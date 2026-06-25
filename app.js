/**
 * app.js – AV-SynthRestore 3D
 * Three.js wireframe node-graph with neon particle streams driven by
 * live WebSocket telemetry from the FastAPI backend.
 *
 * Architecture
 * ────────────
 * Scene: IcosahedronGeometry wireframe nodes + CatmullRomCurve3 tube edges
 *        + BufferGeometry Points particle streams.
 * Telemetry: WebSocket JSON → activates edges, updates glow intensity, moves
 *            UI progress bars and readout cells.
 * REST: Upload → /api/upload, Process → /api/process/{id}, Download → /api/download/{id}
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Constants ──────────────────────────────────────────────────────────────────

const API_BASE = (window.location.protocol.startsWith('http')) 
  ? window.location.origin 
  : 'http://localhost:8765';

const WS_BASE = (window.location.protocol.startsWith('http'))
  ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`
  : 'ws://localhost:8765';

// Node definitions: id → { pos:[x,y,z], color:hex, label:str, labelColor:cssVar }
const NODE_DEFS = {
  input:   { pos: [-11,  0,  0], color: 0x00e676, label: 'INPUT FILE',   css: '--c-green'  },
  demuxer: { pos: [ -4,  0,  0], color: 0x00e5ff, label: 'DEMUXER',      css: '--c-cyan'   },
  audio:   { pos: [  3,  4,  1.5], color: 0xff8c00, label: 'AUDIO REPAIR', css: '--c-amber'  },
  video:   { pos: [  3, -4, -1.5], color: 0xb400ff, label: '4K UPSCALER',  css: '--c-violet' },
  remuxer: { pos: [ 10,  0,  0], color: 0x00e5ff, label: 'REMUXER',      css: '--c-cyan'   },
  output:  { pos: [ 17,  0,  0], color: 0x00e676, label: 'OUTPUT FILE',  css: '--c-green'  },
};

// Edge definitions: id → { from, to, color:hex, particleColor:hex }
const EDGE_DEFS = [
  { id: 'e_in_demux',    from: 'input',   to: 'demuxer', color: 0x00e676, pColor: 0x00e676 },
  { id: 'e_demux_audio', from: 'demuxer', to: 'audio',   color: 0xff8c00, pColor: 0xff8c00 },
  { id: 'e_demux_video', from: 'demuxer', to: 'video',   color: 0xb400ff, pColor: 0xb400ff },
  { id: 'e_audio_remux', from: 'audio',   to: 'remuxer', color: 0xff8c00, pColor: 0xff8c00 },
  { id: 'e_video_remux', from: 'video',   to: 'remuxer', color: 0xb400ff, pColor: 0xb400ff },
  { id: 'e_remux_out',   from: 'remuxer', to: 'output',  color: 0x00e676, pColor: 0x00e676 },
];

// Stage name → which edges should be active (receiving particles)
const STAGE_EDGES = {
  pipeline_start:           [],
  input_loaded:             [],
  probed:                   [],
  demuxing:                 ['e_in_demux'],
  demuxed:                  ['e_in_demux'],
  frame_extraction:         ['e_in_demux'],
  frames_extracted:         ['e_in_demux'],
  parallel_processing_start:['e_demux_audio', 'e_demux_video'],
  audio_load:               ['e_demux_audio'],
  audio_loaded:             ['e_demux_audio'],
  gaps_detected:            ['e_demux_audio'],
  audio_inpainting:         ['e_demux_audio', 'e_audio_remux'],
  audio_inpainted:          ['e_demux_audio', 'e_audio_remux'],
  noise_reduction:          ['e_demux_audio', 'e_audio_remux'],
  audio_saving:             ['e_audio_remux'],
  audio_complete:           ['e_audio_remux'],
  video_init:               ['e_demux_video'],
  video_upscaling:          ['e_demux_video', 'e_video_remux'],
  video_complete:           ['e_video_remux'],
  processing_complete:      ['e_audio_remux', 'e_video_remux'],
  remuxing:                 ['e_audio_remux', 'e_video_remux', 'e_remux_out'],
  pipeline_complete:        ['e_remux_out'],
  error:                    [],
};

// ── Scene State ────────────────────────────────────────────────────────────────

const S = {
  // Three.js core
  renderer:  null,
  scene:     null,
  camera:    null,
  controls:  null,
  clock:     new THREE.Clock(),
  // Scene objects
  nodes:    {},    // id → NodeObject
  streams:  {},    // edgeId → ParticleStream
  starField: null,
  // Job / WS
  jobId:    null,
  ws:       null,
  uploadedFile: null,
  // Telemetry cache
  tel: {
    stage: 'idle', overall_progress: 0,
    audio_progress: 0, video_progress: 0,
    audio_gap_filled_pct: 0, gap_count: 0,
    frame: 0, total_frames: 0,
    gpu_util: 0, gpu_temp: 0, fps: 0,
    elapsed_seconds: 0, output_size_mb: 0,
    device: '—', upscaler: '—',
  },
  // Elapsed timer
  _startTs: null,
  _elapsedInterval: null,
};


// ── ParticleStream Class ───────────────────────────────────────────────────────

class ParticleStream {
  /**
   * A set of particles travelling along a CatmullRomCurve3.
   * Uses BufferGeometry Points with additive blending for a neon glow effect.
   *
   * @param {THREE.Scene}            scene   – Parent scene.
   * @param {THREE.CatmullRomCurve3} curve   – Path the particles follow.
   * @param {number}                 color   – Hex colour (e.g. 0x00e5ff).
   * @param {number}                 count   – Number of particles.
   */
  constructor(scene, curve, color, count = 90) {
    this.curve  = curve;
    this.count  = count;
    this.scene  = scene;
    this.active = false;

    // Smooth interpolation state
    this._opacity      = 0;
    this._targetOpacity = 0;
    this._speed        = 0.00035;
    this._targetSpeed  = 0.00035;

    // Spread phases evenly so particles are distributed along the full curve
    this.phases = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      this.phases[i] = i / count;
    }

    // BufferGeometry with position attribute
    const positions = new Float32Array(count * 3);
    const geometry  = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Material: additive blending creates the glow-on-dark look
    const material = new THREE.PointsMaterial({
      color:       new THREE.Color(color),
      size:        0.22,
      transparent: true,
      opacity:     0,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false,
      sizeAttenuation: true,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.renderOrder = 2;
    scene.add(this.points);

    this._syncPositions();
  }

  /** Call once per frame with delta time in milliseconds. */
  update(deltaMs) {
    // Smooth opacity towards target
    this._opacity += (this._targetOpacity - this._opacity) * 0.06;
    this.points.material.opacity = this._opacity;

    if (this._opacity < 0.004) return;   // skip position work when invisible

    // Smooth speed towards target
    this._speed += (this._targetSpeed - this._speed) * 0.04;

    const adv = this._speed * deltaMs;
    for (let i = 0; i < this.count; i++) {
      this.phases[i] = (this.phases[i] + adv) % 1.0;
    }
    this._syncPositions();
  }

  /** Push world-space positions from phases onto the geometry buffer. */
  _syncPositions() {
    const pos = this.points.geometry.attributes.position.array;
    for (let i = 0; i < this.count; i++) {
      const t  = this.phases[i];
      const pt = this.curve.getPointAt(t);
      pos[i * 3]     = pt.x;
      pos[i * 3 + 1] = pt.y;
      pos[i * 3 + 2] = pt.z;
    }
    this.points.geometry.attributes.position.needsUpdate = true;
  }

  /** Activate/deactivate the stream (fades in / out and adjusts speed). */
  setActive(active, speedMultiplier = 1.0) {
    this.active          = active;
    this._targetOpacity  = active ? 0.88 : 0;
    this._targetSpeed    = active ? 0.0018 * speedMultiplier : 0.0003;
  }

  /** Override speed multiplier while already active (e.g. near completion). */
  setSpeedMultiplier(m) {
    if (this.active) this._targetSpeed = 0.0018 * m;
  }

  dispose() {
    this.scene.remove(this.points);
    this.points.geometry.dispose();
    this.points.material.dispose();
  }
}


// ── Scene Initialisation ───────────────────────────────────────────────────────

function initScene() {
  const wrap   = document.getElementById('canvas-wrap');
  const canvas = document.getElementById('three-canvas');
  const W      = wrap.clientWidth;
  const H      = wrap.clientHeight;

  // Renderer
  S.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  S.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  S.renderer.setSize(W, H);
  S.renderer.setClearColor(0x020a12, 1);
  S.renderer.shadowMap.enabled = false;

  // Scene
  S.scene = new THREE.Scene();
  S.scene.fog = new THREE.FogExp2(0x020a12, 0.018);

  // Camera
  S.camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 500);
  S.camera.position.set(3, 4, 22);
  S.camera.lookAt(3, 0, 0);

  // Orbit controls
  S.controls = new OrbitControls(S.camera, canvas);
  S.controls.enableDamping   = true;
  S.controls.dampingFactor   = 0.06;
  S.controls.minDistance     = 8;
  S.controls.maxDistance     = 60;
  S.controls.maxPolarAngle   = Math.PI * 0.72;
  S.controls.target.set(3, 0, 0);

  // Ambient fill
  const ambient = new THREE.AmbientLight(0x001122, 0.4);
  S.scene.add(ambient);

  // Build the graph
  _buildStarField();
  _buildGrid();
  _buildNodes();
  _buildEdges();
  _buildNodeLabels();

  // Resize handler
  window.addEventListener('resize', _onResize);
}

function _onResize() {
  const wrap = document.getElementById('canvas-wrap');
  const W    = wrap.clientWidth;
  const H    = wrap.clientHeight;
  S.camera.aspect = W / H;
  S.camera.updateProjectionMatrix();
  S.renderer.setSize(W, H);
}


// ── Star Field ─────────────────────────────────────────────────────────────────

function _buildStarField() {
  const STAR_COUNT = 5000;
  const positions  = new Float32Array(STAR_COUNT * 3);
  const SPREAD     = 300;

  for (let i = 0; i < STAR_COUNT; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * SPREAD;
    positions[i * 3 + 1] = (Math.random() - 0.5) * SPREAD;
    positions[i * 3 + 2] = (Math.random() - 0.5) * SPREAD;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const mat = new THREE.PointsMaterial({
    color: 0x4488aa, size: 0.12,
    transparent: true, opacity: 0.45,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });

  S.starField = new THREE.Points(geo, mat);
  S.scene.add(S.starField);
}


// ── Grid ───────────────────────────────────────────────────────────────────────

function _buildGrid() {
  const grid = new THREE.GridHelper(80, 40, 0x0d2137, 0x0a1828);
  grid.position.y = -6;
  grid.material.transparent = true;
  grid.material.opacity     = 0.5;
  S.scene.add(grid);
}


// ── Nodes ──────────────────────────────────────────────────────────────────────

function _buildNodes() {
  for (const [id, def] of Object.entries(NODE_DEFS)) {
    S.nodes[id] = _createNode(id, def);
  }
}

function _createNode(id, def) {
  const group = new THREE.Group();
  group.position.set(...def.pos);

  const c = new THREE.Color(def.color);

  // ── Solid inner icosahedron core (very dim)
  const coreGeo = new THREE.IcosahedronGeometry(0.28, 1);
  const coreMat = new THREE.MeshBasicMaterial({
    color: c, transparent: true, opacity: 0.06,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const core = new THREE.Mesh(coreGeo, coreMat);
  group.add(core);

  // ── Primary wireframe icosahedron
  const wireGeo = new THREE.IcosahedronGeometry(0.55, 2);
  const wireMat = new THREE.MeshBasicMaterial({
    color: c, wireframe: true,
    transparent: true, opacity: 0.75,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const wire = new THREE.Mesh(wireGeo, wireMat);
  group.add(wire);

  // ── Inner glow icosahedron (slightly larger, lower opacity)
  const igGeo = new THREE.IcosahedronGeometry(0.7, 1);
  const igMat = new THREE.MeshBasicMaterial({
    color: c, transparent: true, opacity: 0.04,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.FrontSide,
  });
  const innerGlow = new THREE.Mesh(igGeo, igMat);
  group.add(innerGlow);

  // ── Outer glow spheres (the "halo" effect without post-processing)
  const glows = [];
  [1.1, 1.6, 2.3].forEach((r, i) => {
    const g = new THREE.Mesh(
      new THREE.SphereGeometry(r, 8, 6),
      new THREE.MeshBasicMaterial({
        color: c, transparent: true,
        opacity: [0.05, 0.025, 0.01][i],
        blending: THREE.AdditiveBlending, depthWrite: false,
        side: THREE.FrontSide,
      })
    );
    glows.push(g);
    group.add(g);
  });

  // ── Equatorial orbit ring
  const ringGeo = new THREE.TorusGeometry(0.72, 0.012, 4, 64);
  const ringMat = new THREE.MeshBasicMaterial({
    color: c, transparent: true, opacity: 0.45,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  S.scene.add(group);

  return { group, core, wire, innerGlow, glows, ring, wireMat, coreMat, igMat, ringMat, color: c, _baseOpacity: 0.75, _glowScale: 1.0 };
}


// ── Edges & Particle Streams ───────────────────────────────────────────────────

function _buildEdges() {
  for (const def of EDGE_DEFS) {
    _createEdge(def);
  }
}

function _createEdge(def) {
  const fromPos = new THREE.Vector3(...NODE_DEFS[def.from].pos);
  const toPos   = new THREE.Vector3(...NODE_DEFS[def.to].pos);

  // Control points for a smooth arch between nodes
  const mid = fromPos.clone().lerp(toPos, 0.5);
  // Perpendicular offset to create a visible arc (especially for split paths)
  const dx   = toPos.x - fromPos.x;
  const dy   = toPos.y - fromPos.y;
  const perp = new THREE.Vector3(-dy, dx, 0).normalize();
  const archHeight = Math.min(fromPos.distanceTo(toPos) * 0.18, 2.0);
  mid.add(perp.clone().multiplyScalar(archHeight * 0.5));
  mid.z += (Math.random() - 0.5) * 0.8;  // slight Z variation for 3-D depth

  const curve = new THREE.CatmullRomCurve3([fromPos, mid, toPos], false, 'catmullrom', 0.5);

  // ── Tube geometry for the edge line
  const tubeGeo = new THREE.TubeGeometry(curve, 48, 0.018, 4, false);
  const tubeMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(def.color),
    transparent: true,
    opacity: 0.18,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const tube = new THREE.Mesh(tubeGeo, tubeMat);
  tube.renderOrder = 1;
  S.scene.add(tube);

  // ── Particle stream along the same curve
  S.streams[def.id] = new ParticleStream(S.scene, curve, def.pColor, 90);

  return { tube, curve };
}


// ── Node Labels (HTML overlay) ─────────────────────────────────────────────────

function _buildNodeLabels() {
  const container = document.getElementById('node-labels');
  for (const [id, def] of Object.entries(NODE_DEFS)) {
    const el = document.createElement('div');
    el.className = 'node-label';
    el.id        = `lbl-${id}`;
    el.innerHTML = `${def.label}<div class="node-label-pct" id="lbl-pct-${id}"></div>`;
    container.appendChild(el);
  }
}

/** Project 3-D node positions → CSS pixel coords and update label positions. */
function _updateLabelPositions() {
  const wrap = document.getElementById('canvas-wrap');
  const W    = wrap.clientWidth;
  const H    = wrap.clientHeight;

  for (const [id, node] of Object.entries(S.nodes)) {
    const el = document.getElementById(`lbl-${id}`);
    if (!el) continue;

    // Get the node's world position
    const worldPos = new THREE.Vector3();
    node.group.getWorldPosition(worldPos);
    // Slightly above the node
    worldPos.y += 1.1;

    // Project to NDC
    const ndc = worldPos.clone().project(S.camera);

    // Convert NDC → CSS pixels
    const x = (ndc.x * 0.5 + 0.5) * W;
    const y = (-ndc.y * 0.5 + 0.5) * H;

    // Hide if behind camera or off-screen
    if (ndc.z > 1 || x < -100 || x > W + 100 || y < -50 || y > H + 50) {
      el.style.display = 'none';
    } else {
      el.style.display = '';
      el.style.left    = `${x}px`;
      el.style.top     = `${y}px`;
    }
  }
}


// ── Animation Loop ─────────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  const deltaMs  = S.clock.getDelta() * 1000;   // milliseconds

  // Orbit controls damping
  S.controls.update();

  const t = S.clock.elapsedTime;

  // ── Animate each node
  for (const [id, node] of Object.entries(S.nodes)) {
    // Slow continuous rotation of the wireframe shell
    node.wire.rotation.y += deltaMs * 0.00022;
    node.wire.rotation.x += deltaMs * 0.00012;

    // Ring counter-rotation for visual interest
    node.ring.rotation.z += deltaMs * 0.00018;

    // Pulsing glow scale (sinusoidal "heartbeat")
    const pulse = 1.0 + Math.sin(t * 1.8 + _nodePhaseOffset(id)) * 0.06 * node._glowScale;
    node.glows[0].scale.setScalar(pulse);
    node.glows[1].scale.setScalar(pulse * 0.97);
    node.glows[2].scale.setScalar(pulse * 0.94);
  }

  // ── Update particle streams
  for (const stream of Object.values(S.streams)) {
    stream.update(deltaMs);
  }

  // ── Slow drift of star field
  if (S.starField) {
    S.starField.rotation.y += deltaMs * 0.000018;
    S.starField.rotation.x += deltaMs * 0.000008;
  }

  // ── Sync HTML label positions
  _updateLabelPositions();

  S.renderer.render(S.scene, S.camera);
}

function _nodePhaseOffset(id) {
  const offsets = { input: 0, demuxer: 1.0, audio: 2.1, video: 3.2, remuxer: 4.3, output: 5.4 };
  return offsets[id] || 0;
}


// ── Telemetry → Scene Updates ─────────────────────────────────────────────────

/**
 * Apply a telemetry payload from the backend WebSocket to the scene state.
 * This is the central bridge between the Python pipeline and the 3-D GUI.
 */
function applyTelemetry(data) {
  const stage = data.stage || '';

  // Merge into local telemetry cache
  Object.assign(S.tel, {
    stage:               stage,
    overall_progress:    data.overall_progress    ?? S.tel.overall_progress,
    audio_progress:      data.audio_progress      ?? S.tel.audio_progress,
    video_progress:      data.video_progress      ?? S.tel.video_progress,
    audio_gap_filled_pct:data.audio_gap_filled_pct ?? S.tel.audio_gap_filled_pct,
    gap_count:           data.gap_count           ?? S.tel.gap_count,
    frame:               data.frame               ?? S.tel.frame,
    total_frames:        data.total_frames        ?? S.tel.total_frames,
    gpu_util:            data.gpu_util            ?? S.tel.gpu_util,
    gpu_temp:            data.gpu_temp            ?? S.tel.gpu_temp,
    fps:                 data.fps                 ?? S.tel.fps,
    elapsed_seconds:     data.elapsed_seconds     ?? S.tel.elapsed_seconds,
    output_size_mb:      data.output_size_mb      ?? S.tel.output_size_mb,
    device:              data.device              ?? S.tel.device,
    upscaler:            data.upscaler            ?? S.tel.upscaler,
  });

  // ── Activate correct particle streams for this stage
  const activeEdges = new Set(STAGE_EDGES[stage] || []);
  for (const [id, stream] of Object.entries(S.streams)) {
    stream.setActive(activeEdges.has(id));
  }

  // ── Node glow intensity based on activity
  _updateNodeGlows(stage, data.branch);

  // ── Update all UI elements
  _updateUI(stage, data);

  // ── Log the event
  _logEvent(stage, data);
}

function _updateNodeGlows(stage, branch) {
  // Map stages to which nodes are "hot"
  const hotNodes = {
    demuxing:                 ['input', 'demuxer'],
    frame_extraction:         ['input', 'demuxer'],
    demuxed:                  ['demuxer'],
    parallel_processing_start:['demuxer', 'audio', 'video'],
    audio_inpainting:         ['audio'],
    audio_complete:           ['audio', 'remuxer'],
    video_upscaling:          ['video'],
    video_complete:           ['video', 'remuxer'],
    processing_complete:      ['audio', 'video', 'remuxer'],
    remuxing:                 ['remuxer'],
    pipeline_complete:        ['remuxer', 'output'],
  };

  const hot = new Set(hotNodes[stage] || []);

  for (const [id, node] of Object.entries(S.nodes)) {
    const isHot    = hot.has(id);
    const isDone   = stage === 'pipeline_complete';

    // Glow scale drives the heartbeat amplitude in animate()
    node._glowScale = isHot ? 3.2 : 1.0;

    // Wireframe opacity
    const targetOpacity = isHot ? 1.0 : (isDone ? 0.9 : 0.38);
    node.wireMat.opacity = THREE.MathUtils.lerp(node.wireMat.opacity, targetOpacity, 0.12);

    // Glow sphere opacities
    node.glows[0].material.opacity = isHot ? 0.18 : 0.04;
    node.glows[1].material.opacity = isHot ? 0.09 : 0.02;
    node.glows[2].material.opacity = isHot ? 0.04 : 0.008;

    // Node label state classes
    const lbl = document.getElementById(`lbl-${id}`);
    if (lbl) {
      lbl.className = 'node-label';
      if (isDone) {
        lbl.classList.add('complete');
      } else if (isHot) {
        lbl.classList.add(
          branch === 'audio' ? 'audio-active' :
          branch === 'video' ? 'video-active' : 'active'
        );
      }
    }
  }

  // Per-branch percentage display on labels
  _setLabelPct('audio', S.tel.audio_progress);
  _setLabelPct('video', S.tel.video_progress);
  _setLabelPct('remuxer',
    stage === 'pipeline_complete' ? 100 :
    stage === 'remuxing'          ? 50  : null
  );
}

function _setLabelPct(nodeId, pct) {
  const el = document.getElementById(`lbl-pct-${nodeId}`);
  if (el) el.textContent = pct != null ? `${Math.round(pct)}%` : '';
}


// ── UI Update ─────────────────────────────────────────────────────────────────

function _updateUI(stage, data) {
  const tel = S.tel;

  // ── Footer bar
  const pct = Math.max(0, Math.min(100, tel.overall_progress));
  _setBarAndText('overall-bar-fill', 'overall-pct', pct,
    stage === 'pipeline_complete' ? 'complete' : '');
  _setBarAndText('', 'tel-overall', pct, 'cyan');

  // ── Footer stage text
  document.getElementById('stage-text').textContent = _fmtStage(stage);

  // ── Compute meters (right panel)
  const gpuPct = tel.gpu_util;
  _setMeter('bar-gpu',  'tel-gpu',  gpuPct, '', '%');

  const tempPct = Math.min(100, (tel.gpu_temp / 100) * 100);
  _setMeter('bar-temp', 'tel-temp', tempPct, 'amber', `${tel.gpu_temp}°C`);
  document.getElementById('hdr-gpu').textContent  = gpuPct > 0 ? `${gpuPct}%`         : '—';
  document.getElementById('hdr-temp').textContent = tel.gpu_temp > 0 ? `${tel.gpu_temp}°C` : '—';

  // ── Audio / video branch bars
  _setMeter('bar-audio', 'tel-audio', tel.audio_progress, 'amber',  `${Math.round(tel.audio_progress)}%`);
  _setMeter('bar-video', 'tel-video', tel.video_progress, 'violet', `${Math.round(tel.video_progress)}%`);

  // ── Gaps filled bar
  const gapsPct = tel.gap_count > 0
    ? (tel.audio_gap_filled_pct)
    : (stage === 'audio_complete' ? 100 : 0);
  const gapsText = tel.gap_count > 0
    ? `${Math.round(tel.audio_gap_filled_pct)}% (${tel.gap_count} gaps)`
    : '—';
  _setMeter('bar-gaps', 'tel-gaps', gapsPct, 'amber', gapsText);

  // ── Readout cells
  document.getElementById('rout-frame').textContent = tel.frame > 0 ? tel.frame : '—';
  document.getElementById('rout-total').textContent = tel.total_frames > 0 ? tel.total_frames : '—';
  document.getElementById('rout-fps').textContent   = tel.fps > 0 ? tel.fps.toFixed(1) : '—';
  document.getElementById('rout-batch').textContent = data.batch != null ? `${data.batch}/${data.total_batches ?? '?'}` : '—';

  // ── Session KV
  if (tel.output_size_mb > 0) document.getElementById('kv-size').textContent = `${tel.output_size_mb} MB`;
  if (tel.device && tel.device !== '—') {
    document.getElementById('kv-device').textContent     = tel.device.toUpperCase();
    document.getElementById('cfg-device').textContent    = tel.device.toUpperCase();
  }
  if (tel.upscaler && tel.upscaler !== '—') {
    document.getElementById('kv-upscaler-used').textContent = tel.upscaler;
  }
  if (data.audio_gaps_repaired != null) {
    document.getElementById('tel-gaps').textContent = `${data.audio_gaps_repaired} fixed`;
  }

  // ── Stage indicator list (left panel)
  _updateStageList(stage, tel.audio_progress, tel.video_progress);

  // ── Download button enable on completion
  if (stage === 'pipeline_complete') {
    document.getElementById('btn-download').disabled = false;
    document.getElementById('btn-process').classList.remove('running');
    document.getElementById('btn-process').disabled = true;
    _stopElapsedTimer();
  }
  if (stage === 'error') {
    document.getElementById('btn-process').classList.remove('running');
    document.getElementById('btn-process').disabled = false;
    _stopElapsedTimer();
  }
}

function _setMeter(barId, labelId, pct, colorClass, text) {
  if (barId) {
    const bar = document.getElementById(barId);
    if (bar) {
      bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
      if (colorClass) {
        bar.className = `tel-bar-fill ${colorClass}`;
      }
    }
  }
  if (labelId) {
    const lbl = document.getElementById(labelId);
    if (lbl) lbl.textContent = text || `${Math.round(pct)}%`;
  }
}

function _setBarAndText(barId, textId, pct, extra) {
  if (barId) {
    const bar = document.getElementById(barId);
    if (bar) {
      bar.style.width = `${pct}%`;
      if (extra) bar.className = `tel-bar-fill ${extra}`;
    }
  }
  if (textId) {
    const txt = document.getElementById(textId);
    if (txt) {
      txt.textContent = `${Math.round(pct)}%`;
      if (extra) txt.className = `tel-value ${extra}`;
    }
  }
}

function _updateStageList(stage, audioPct, videoPct) {
  const stateMap = {
    input_loaded:             { ingest: 'active' },
    probed:                   { ingest: 'complete' },
    demuxing:                 { ingest: 'complete', demux: 'active' },
    demuxed:                  { ingest: 'complete', demux: 'active' },
    frames_extracted:         { ingest: 'complete', demux: 'complete' },
    parallel_processing_start:{ ingest: 'complete', demux: 'complete', audio: 'active', video: 'active' },
    audio_inpainting:         { ingest: 'complete', demux: 'complete', audio: 'active' },
    audio_complete:           { ingest: 'complete', demux: 'complete', audio: 'complete' },
    video_upscaling:          { ingest: 'complete', demux: 'complete', video: 'active' },
    video_complete:           { ingest: 'complete', demux: 'complete', video: 'complete' },
    processing_complete:      { ingest: 'complete', demux: 'complete', audio: 'complete', video: 'complete' },
    remuxing:                 { ingest: 'complete', demux: 'complete', audio: 'complete', video: 'complete', remux: 'active' },
    pipeline_complete:        { ingest: 'complete', demux: 'complete', audio: 'complete', video: 'complete', remux: 'complete', output: 'complete' },
    error:                    {},
  };
  const state = stateMap[stage] || {};

  const map = { ingest: 'stg-ingest', demux: 'stg-demux', audio: 'stg-audio', video: 'stg-video', remux: 'stg-remux', output: 'stg-output' };
  const pcts = { ingest: '✓', demux: '✓', audio: `${Math.round(audioPct)}%`, video: `${Math.round(videoPct)}%`, remux: '…', output: '✓' };

  for (const [key, elId] of Object.entries(map)) {
    const el    = document.getElementById(elId);
    const pctEl = document.getElementById(`${elId}-pct`);
    if (!el) continue;
    const s = state[key];
    el.className = s ? `stage-item ${s}` : 'stage-item';
    if (pctEl && s) pctEl.textContent = s === 'complete' ? '✓' : (s === 'active' ? pcts[key] : '—');
  }
}

function _fmtStage(stage) {
  return (stage || 'idle')
    .replace(/_/g, ' ')
    .toUpperCase();
}


// ── Elapsed Timer ──────────────────────────────────────────────────────────────

function _startElapsedTimer() {
  S._startTs = Date.now();
  S._elapsedInterval = setInterval(() => {
    const secs = Math.floor((Date.now() - S._startTs) / 1000);
    const t    = _fmtTime(secs);
    document.getElementById('kv-elapsed').textContent  = t;
    document.getElementById('footer-elapsed').textContent = t;
  }, 1000);
}

function _stopElapsedTimer() {
  if (S._elapsedInterval) {
    clearInterval(S._elapsedInterval);
    S._elapsedInterval = null;
  }
}

function _fmtTime(secs) {
  const h = String(Math.floor(secs / 3600)).padStart(2, '0');
  const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}


// ── Event Log ─────────────────────────────────────────────────────────────────

function _logEvent(stage, data) {
  const box = document.getElementById('log-box');
  if (!box) return;

  const ts  = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const cls = stage === 'error' ? 'error' :
              stage === 'pipeline_complete' ? 'ok' :
              stage.includes('complete') ? 'ok' : 'info';

  const entry = document.createElement('div');
  entry.className = `log-entry ${cls}`;
  const stageText = _fmtStage(stage);
  let extra = '';
  if (data.error) extra = ` – ${data.error}`;
  else if (data.frame > 0) extra = ` f${data.frame}`;
  entry.innerHTML = `<span class="log-ts">${ts}</span>${stageText}${extra}`;

  box.appendChild(entry);
  box.scrollTop = box.scrollHeight;

  // Keep max 120 entries
  while (box.children.length > 120) {
    box.removeChild(box.firstChild);
  }
}

function uiLog(msg, cls = '') {
  const box = document.getElementById('log-box');
  if (!box) return;
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const el = document.createElement('div');
  el.className = `log-entry ${cls}`;
  el.innerHTML = `<span class="log-ts">${ts}</span>${msg}`;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}


// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS(jobId) {
  if (S.ws) {
    try { S.ws.close(); } catch (_) {}
  }

  S.ws = new WebSocket(`${WS_BASE}/ws/${jobId}`);

  S.ws.onopen = () => {
    _setWsStatus('connected');
    uiLog('WebSocket connected', 'info');
    // Send keepalive pings
    S._wsPingInterval = setInterval(() => {
      if (S.ws && S.ws.readyState === WebSocket.OPEN) {
        S.ws.send('ping');
      }
    }, 20000);
  };

  S.ws.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); }
    catch { return; }

    if (data.type === 'keepalive' || data.type === 'pong' || data.type === 'connected') return;

    applyTelemetry(data);
  };

  S.ws.onerror = (e) => {
    _setWsStatus('error');
    uiLog('WebSocket error', 'error');
  };

  S.ws.onclose = () => {
    _setWsStatus('disconnected');
    clearInterval(S._wsPingInterval);
    uiLog('WebSocket closed', 'warn');
  };
}

function _setWsStatus(state) {
  const dot   = document.getElementById('ws-dot');
  const label = document.getElementById('ws-label');
  dot.className   = `ws-dot ${state === 'connected' ? 'connected' : state === 'error' ? 'error' : ''}`;
  label.textContent = state.toUpperCase();
}


// ── REST API Calls ─────────────────────────────────────────────────────────────

async function uploadFile() {
  if (!S.uploadedFile) return;

  const btn = document.getElementById('btn-upload');
  btn.disabled = true;
  btn.textContent = '⬆ Uploading…';
  uiLog(`Uploading ${S.uploadedFile.name}…`);

  const form = new FormData();
  form.append('file', S.uploadedFile);

  try {
    const res  = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    S.jobId = data.job_id;
    document.getElementById('hdr-job-id').textContent =
      data.job_id.substring(0, 8) + '…';
    uiLog(`Upload OK – job ${data.job_id.substring(0, 8)}… (${data.size_mb} MB)`, 'ok');

    // Open WS connection immediately so we don't miss early events
    connectWS(S.jobId);

    btn.textContent = '✓ Uploaded';
    document.getElementById('btn-process').disabled = false;

  } catch (err) {
    uiLog(`Upload failed: ${err.message}`, 'error');
    btn.disabled = false;
    btn.textContent = '⬆ Retry Upload';
  }
}

async function startProcessing() {
  if (!S.jobId) return;

  const btn = document.getElementById('btn-process');
  btn.disabled = true;
  btn.classList.add('running');
  btn.textContent = '⏳ Processing…';

  _startElapsedTimer();
  uiLog('Starting restoration pipeline…', 'info');

  try {
    const res  = await fetch(`${API_BASE}/api/process/${S.jobId}`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    uiLog(`Pipeline started: ${data.status}`, 'ok');
  } catch (err) {
    uiLog(`Start failed: ${err.message}`, 'error');
    btn.classList.remove('running');
    btn.disabled = false;
    btn.textContent = '▶ Start Restoration';
    _stopElapsedTimer();
  }
}

async function downloadResult() {
  if (!S.jobId) return;
  uiLog('Downloading result…', 'info');

  try {
    const res = await fetch(`${API_BASE}/api/download/${S.jobId}`);
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    const disposition = res.headers.get('Content-Disposition') || '';
    const nameMatch   = disposition.match(/filename="?([^"]+)"?/);
    a.href     = url;
    a.download = nameMatch ? nameMatch[1] : `restored_${S.jobId.substring(0,8)}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    uiLog('Download started', 'ok');
  } catch (err) {
    uiLog(`Download failed: ${err.message}`, 'error');
  }
}


// ── File Drop / Select ─────────────────────────────────────────────────────────

function _setupFileHandlers() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');

  // Click to browse
  fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) _handleFileSelect(e.target.files[0]);
  });

  // Drag and drop
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) _handleFileSelect(e.dataTransfer.files[0]);
  });

  // Button wiring
  document.getElementById('btn-upload').addEventListener('click', uploadFile);
  document.getElementById('btn-process').addEventListener('click', startProcessing);
  document.getElementById('btn-download').addEventListener('click', downloadResult);
}

function _handleFileSelect(file) {
  S.uploadedFile = file;
  const nameEl = document.getElementById('file-name');
  const sizeKB = (file.size / 1024).toFixed(0);
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  nameEl.textContent = `${file.name}  (${sizeMB} MB)`;
  document.getElementById('btn-upload').disabled = false;
  uiLog(`Selected: ${file.name} (${sizeKB} KB)`);
}


// ── Initialisation ─────────────────────────────────────────────────────────────

function init() {
  initScene();
  _setupFileHandlers();
  animate();
  uiLog('AV-SynthRestore 3D ready. Select a video file to begin.', 'ok');

  // Fetch config to populate UI
  fetch(`${API_BASE}/api/config`)
    .then(r => r.json())
    .then(cfg => {
      if (cfg.video) {
        document.getElementById('cfg-batch').textContent   = `${cfg.video.batch_size || 4} frames`;
        document.getElementById('cfg-upscaler').textContent = cfg.video.upscale_model || 'RealESRGAN ×4';
      }
    })
    .catch(() => {}); // non-critical
}

document.addEventListener('DOMContentLoaded', init);
