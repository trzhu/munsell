import * as THREE from "three";
import { OrbitControls } from "OrbitControls";

// globals
let scene, camera, renderer, controls;
let slicer;
let isPaused = false;

// Y scaling factor so value 0-10 looks bigger
const Y_SCALE = 3;

// dictionary of meshes
// keys: "shell", "pointcloud", "pointcloud_original, cutSurfaces"
// todo: maybe just store the meshes instead of the whole meshobj. i dont bother with any of the other fields anyways
const meshes = {};

// Scene configurations
const sceneConfigs = {
  default: {
    name: "Volume",
    visible: ["shell", "cutSurfaces"],
    hidden: ["pointcloud_interpolated", "pointcloud_original"],
  },
  pointCloud: {
    name: "Points",
    visible: ["pointcloud_original"],
    hidden: ["shell", "pointcloud_interpolated"],
  },
  debug: {
    name: "Debug",
    // visible: ["debug"],
    visible: ["shell", "cutSurfaces"],
    // visible: ["pointcloud_interpolated"],
    hidden: ["shell", "pointcloud_original"],
  },
};

// init scene + camera + lights
function initScene() {
  // scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x7f7f7f);

  // init camera
  const container = document.getElementById("render-container");
  const aspect = container.clientWidth / container.clientHeight;
  // orthographic camera setup
  // parameters dont matter bc we're gonna centre it to our mesh later anyways
  camera = new THREE.OrthographicCamera(
    -1 * aspect,
    1 * aspect,
    1,
    -1,
    0.1,
    5000
  );

  // init and configurerenderer
  renderer = new THREE.WebGLRenderer({
    stencil: true,
    antialias: true,
    alpha: false,
  });
  renderer.localClippingEnabled = true;

  // handle clearing manually (needed for stencil buffer)
  renderer.autoClear = false;
  renderer.setClearColor(0x000000, 1.0);
  container.appendChild(renderer.domElement);

  // controls
  controls = new OrbitControls(camera, renderer.domElement);
  // TODO: might need custom pan controls if I want the panning behaviour i want
  // controls.enablePan = false;

  // init lights
  const light = new THREE.DirectionalLight(0xffffff, 3.5);
  light.position.set(5, 5, 5);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xeeeeee));
}

function initUI() {
  // BUTTONS
  // play/pause button
  const pauseButton = document.getElementById("toggle-rotation");
  pauseButton.addEventListener("click", () => {
    isPaused = !isPaused;
    pauseButton.textContent = isPaused ? "Play Rotation" : "Pause Rotation";
  });

  // lighting toggle button
  const toggleLightButton = document.getElementById("toggle-light");
  toggleLightButton.addEventListener("click", () => {
    slicer.toggleLighting();
    if (slicer.uniforms.useLighting.value < 0.5) {
      toggleLightButton.textContent = "Turn on Lighting";
    } else {
      toggleLightButton.textContent = "Show Exact Color";
    }
  });

  const toggleRGBButton = document.getElementById("toggle-rgb");
  toggleRGBButton.addEventListener("click", () => {
    slicer.toggleRGB();
    if (slicer.uniforms.showOutsideRGB.value > 0.5) {
      toggleRGBButton.textContent = "Clip to RGB limits";
    } else {
      toggleRGBButton.textContent = "Show all colours";
    }
  });

  /********** SLIDERS ************/
  // circular hue slider
  const circularHueSlider = new CircularSlider("hue-slider");
  circularHueSlider.onChange = (range) => {
    slicer.setHueRange(range.start, range.end);
  };
  circularHueSlider.onChange(circularHueSlider.getHueRange());

  // two-handle linear value slider
  const valueSlider = new TwoHandleSlider("value-slider", 0, 10);
  valueSlider.onChange = (range) => {
    slicer.setValueRange(range.start, range.end);
  };
  valueSlider.onChange(valueSlider.getValues());

  // two-handle linear chroma slider
  const chromaSlider = new TwoHandleSlider("chroma-slider", 0, 38);
  chromaSlider.onChange = (range) => {
    slicer.setChromaRange(range.start, range.end);
  };
  // idk initalize it to some random colors
  chromaSlider.setGradient("#808080", "#ff9b00");
  chromaSlider.onChange(chromaSlider.getValues());

  // set scene
  const sceneSelect = document.getElementById("sceneSelect");
  sceneSelect.addEventListener("change", (event) => {
    switchScene(event.target.value);
  });

  switchScene(sceneSelect.value);
}

// Scene switching function
function switchScene(sceneKey) {
  const config = sceneConfigs[sceneKey];

  if (!config) return;

  scene.clear();

  config.visible.forEach((meshName) => {
    if (meshes[meshName] && meshes[meshName].mesh) {
      scene.add(meshes[meshName].mesh);
    }
  });

  // // hide all meshes
  // Object.keys(meshes).forEach((meshName) => {
  //   if (meshes[meshName] && meshes[meshName].mesh) {
  //     meshes[meshName].mesh.visible = false;
  //   }
  // });

  // // show only the meshes specified in the scene config
  // config.visible.forEach((meshName) => {
  //   if (meshes[meshName] && meshes[meshName].mesh) {
  //     meshes[meshName].mesh.visible = true;
  //   }
  // });

  const toggleLightButton = document.getElementById("toggle-light");
  const toggleRGBButton = document.getElementById("toggle-rgb");

  if (sceneKey === "default") {
    toggleLightButton.style.display = "block";
    toggleRGBButton.style.display = "none";
  } else if (sceneKey === "pointCloud") {
    toggleLightButton.style.display = "none";
    toggleRGBButton.style.display = "block";
  }
}

class Slicer {
  constructor() {
    this.uniforms = {
      hueMin: { value: 0.0 },
      hueMax: { value: 2 * Math.PI },
      chromaMin: { value: 0.0 },
      chromaMax: { value: 38.0 },
      valueMin: { value: 0.0 },
      valueMax: { value: 10.0 },
      uSize: { value: 10.0 }, // point size for point clouds
      useLighting: { value: 0.0 },
      showOutsideRGB: { value: 1.0 },
      interiorTexture: { value: null },
    };

    this.shadersPromise = this.loadShaders();
    // group of mesh geometries
    this.group = this.updateCutSurfaces();
    // load max chroma dictionary from jsons
    this.maxChromaPromise = this.loadMaxChromaDict();
    this.maxChroma = null;

    this.maxChromaPromise.then((data) => {
      this.maxChroma = data;
    });

    this.loadTextures();

    // individual mesh references
    this.meshRefs = {
      hueMinPlane: null,
      hueMaxPlane: null,
      valueMinPlane: null,
      valueMaxPlane: null,
      chromaMinCyl: null,
      chromaMaxCyl: null,
    };
  }

  async loadShaders() {
    const [
      meshVertex,
      meshFragment,
      pointsVertex,
      pointsFragment,
      sliceVertex,
      sliceFragment,
    ] = await Promise.all([
      fetch("./shaders/mesh_vertex.glsl").then((r) => r.text()),
      fetch("./shaders/mesh_fragment.glsl").then((r) => r.text()),
      fetch("./shaders/points_vertex.glsl").then((r) => r.text()),
      fetch("./shaders/points_fragment.glsl").then((r) => r.text()),
      fetch("./shaders/slice_vertex.glsl").then((r) => r.text()),
      fetch("./shaders/slice_fragment.glsl").then((r) => r.text()),
    ]);

    return {
      meshVertex,
      meshFragment,
      pointsVertex,
      pointsFragment,
      sliceVertex,
      sliceFragment,
    };
  }

  async loadTextures() {
    const texture3D = await load3DTexture("./texture3d_64.bin", 64);
    this.uniforms.interiorTexture.value = texture3D;
  }

  async getMaterial(type) {
    const shaders = await this.shadersPromise;

    let vertexShader, fragmentShader, side;
    if (type === "points") {
      vertexShader = shaders.pointsVertex;
      fragmentShader = shaders.pointsFragment;
      side = THREE.FrontSide;
    } else if (type === "mesh") {
      vertexShader = shaders.meshVertex;
      fragmentShader = shaders.meshFragment;
      side = THREE.FrontSide;
    } else if (type === "cutSurface") {
      vertexShader = shaders.sliceVertex;
      fragmentShader = shaders.sliceFragment;
      // TODO: will change this to frontside later, only front side is enough as long as the normals point the right way
      side = THREE.DoubleSide;
    } else {
      throw new Error(`Unsupported type: ${type}`);
    }

    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: this.uniforms,
      transparent: true,
      side: side,
    });
  }

  async loadMaxChromaDict() {
    const response = await fetch("max_chroma.json");
    const data = await response.json();

    return Object.fromEntries(
      Object.entries(data).map(([h, values]) => [
        parseFloat(h),
        Object.fromEntries(
          Object.entries(values).map(([v, chroma]) => [parseFloat(v), chroma])
        ),
      ])
    );
  }

  setHueRange(min, max) {
    this.uniforms.hueMin.value = (min * Math.PI) / 180;
    this.uniforms.hueMax.value = (max * Math.PI) / 180;
    this.updateCutSurfaces();
  }

  setChromaRange(min, max) {
    this.uniforms.chromaMin.value = min;
    this.uniforms.chromaMax.value = max;
    this.updateCutSurfaces();
  }

  setValueRange(min, max) {
    this.uniforms.valueMin.value = min;
    this.uniforms.valueMax.value = max;
    this.updateCutSurfaces();
  }

  toggleLighting() {
    this.uniforms.useLighting.value = 1 - this.uniforms.useLighting.value;
  }

  toggleRGB() {
    this.uniforms.showOutsideRGB.value = 1 - this.uniforms.showOutsideRGB.value;
  }

  // updates cut surface positions when min/max h,v,c change
  updateCutSurfaces() {
    if (!this.maxChroma) {
      console.log("maxChroma still loading");
      return;
    }

    // dispose of old surfaces
    if (meshes["cutSurfaces"]) {
      // remove from scene
      scene.remove(meshes["cutSurfaces"].mesh);
      // dispose of both geometries and materials
      meshes["cutSurfaces"].mesh.traverse((child) => {
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((mat) => mat.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      // clear entry from meshes
      delete meshes["cutSurfaces"];
    }

    // references
    Object.keys(this.meshRefs).forEach((key) => {
      this.meshRefs[key] = null;
    });

    const group = new THREE.Group();

    const materialPromise = this.getMaterial("cutSurface");
    materialPromise.then((material) => {

      const huePlanes = this.huePlanes();
      // not ready for these yet
      // const valuePlanes = this.valuePlanes();
      // const chromaCyls = this.chromaCylinders();

      this.meshRefs.hueMinPlane = new THREE.Mesh(huePlanes.minPlane, material);
      this.meshRefs.hueMaxPlane = new THREE.Mesh(huePlanes.maxPlane, material);

      // this.meshRefs.valueMinPlane
      // this.meshRefs.valueMaxPlane etc...

      Object.values(this.meshRefs).forEach((mesh) => {
        if (mesh) {
          group.add(mesh);
        }
      });

      const meshObj = {
        materials: material,
        mesh: group,
        config: "cutSurface",
      };

      scene.add(group);
      meshes["cutSurfaces"] = meshObj;
    });

    return group;
  }

  // TODO:
  // for each cut surface, create geometry for it and hook it up
  // for the first and last vertices of the irregular edge, will need to lerp

  // 2 hue planes (pie slice)
  // max value, min value are always vertices.
  huePlanes() {
    // TEMP: ROUND HUE before lerp is ready
    const hMin =
      9 * Math.round((this.uniforms.hueMin.value * 180) / Math.PI / 9);
    const hMax =
      9 * Math.round((this.uniforms.hueMax.value * 180) / Math.PI / 9);
    // temp as well - later this is only inside the loop
    const vMin = Math.ceil(this.uniforms.valueMin.value);
    const vMax = Math.floor(this.uniforms.valueMax.value);

    const minVertices = [];
    const maxVertices = [];

    console.log("hMin:", hMin, "hMax:", hMax);
    console.log("vMin:", vMin, "vMax:", vMax);

    // todo: add light value grayscale point
    minVertices.push();

    // todo: vertices at start/end hues are interpolated

    // remaining vertices are directly from data
    for (let v = vMin; v < vMax; v += 1) {
      // const cMin = this.maxChroma[hMin]?.[v] || 0;
      // const cMax = this.maxChroma[hMax]?.[v] || 0;

      let cMin, cMax;
      if (!this.maxChroma[hMin]) {
        console.log(`maxChroma[${hMin}] doesn't exist`);
        cMin = 0;
      } else if (this.maxChroma[hMin][v] === undefined) {
        console.log(`maxChroma[${hMin}][${v}] doesn't exist`);
        cMin = 0;
      } else {
        cMin = this.maxChroma[hMin][v];
      }

      if (!this.maxChroma[hMax]) {
        console.log(`maxChroma[${hMax}] doesn't exist`);
        cMax = 0;
      } else if (this.maxChroma[hMax][v] === undefined) {
        console.log(`maxChroma[${hMax}][${v}] doesn't exist`);
        cMax = 0;
      } else {
        cMax = this.maxChroma[hMax][v];
      }

      minVertices.push(this.HVC_to_XYZ(hMin, v, cMin));
      maxVertices.push(this.HVC_to_XYZ(hMax, v, cMax));
    }
    // console.log("min: ", minVertices);
    // console.log("max: ", maxVertices);

    // todo: interpolated end hue
    // todo: add dark value grayscale point

    // convert vertex lists to geometry
    return {
      minPlane: createPolygonGeometry(minVertices),
      maxPlane: createPolygonGeometry(maxVertices, true),
    };
  }

  // 2 value planes (horizontal)
  valuePlanes() {
    const minVertices = [];
    const maxVertices = [];

    // if hue start != hue end, the center vertex must be included
    if (this.uniforms.hueMin.value != this.uniforms.hueMax.value) {
      // minVertices.push()
    }

    // todo

    // return {
    //   minPlane: createPolygonGeometry(minVertices),
    //   maxPlane: createPolygonGeometry(maxVertices, true)
    // }
  }

  chromaCylinders() {
    const minTopVertices = [];
    const minBotVertices = [];
    const maxTopVertices = [];
    const maxBotVertices = [];

    // todo

    return {
      minCyl: createCylinderGeometry(minTopVertices, minBotVertices, false),
      maxCyl: createCylinderGeometry(maxTopVertices, maxBotVertices),
    };
  }

  // cylindrical to cartesian coordinate helpers
  HVC_to_XYZ(h, v, c) {
    const hueRadians = (h * Math.PI) / 180.0;
    const x = c * Math.cos(hueRadians);
    const y = Y_SCALE * v;
    const z = c * Math.sin(hueRadians);

    return { x, y, z };
  }

  XYZ_to_HVC(x, y, z) {
    const v = y / Y_SCALE;
    const c = Math.sqrt(x * x + z * z);
    let h = (Math.atan2(z, x) * 180.0) / Math.PI;
    if (h < 0) {
      h += 360;
    }

    return { h, v, c };
  }

  // todo: might need to account for triangles
  // (the line goes from upper left to lower right)
  lerpMaxChroma(targetHue, targetValue) {}
}

/**
 * creates a planar polygon from vertices arranged in a loop
 * using triangle fan topology (all triangles share the first vertex as a common point)
 * WINDING ORDER: DEFAULTS to ccw (when viewed from the direction of the normal)
 */
function createPolygonGeometry(vertices, reverseWinding = false) {
  console.log("vertices in create polygon geometry: ", vertices);
  const vertexCount = vertices.length;

  if (vertexCount < 3) {
    throw new Error("Need at least 3 vertices to create a loop");
  }

  // flatten vertices array into positions array (expected by THREE)
  const positions = [];
  vertices.forEach((vertex) => {
    positions.push(vertex.x, vertex.y, vertex.z);
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );

  // triangle fan indices (all triangles share vertex 0)
  const indices = [];
  for (let i = 1; i < vertexCount - 1; i++) {
    if (reverseWinding) {
      // CW winding
      indices.push(0, i + 1, i);
    } else {
      // CCW winding
      indices.push(0, i, i + 1);
    }
  }
  // console.log(vertices);
  // console.log(indices);

  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return geometry;
}

function createCylinderGeometry(
  topPositions,
  botPositions,
  normalOutward = true
) {
  // todo
}

class CircularSlider {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.handle1 = document.getElementById("handle1");
    this.handle2 = document.getElementById("handle2");
    this.arcFill = document.getElementById("arc-fill");

    this.centerX = 100;
    this.centerY = 100;
    this.radius = 90;

    this.angle1 = 0;
    this.angle2 = 360;

    this.isDragging = false;
    this.activeHandle = null;
    this.onChange = null;

    this.init();
    this.updateDisplay();
  }

  init() {
    this.handle1.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle1")
    );
    this.handle2.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle2")
    );
    document.addEventListener("mousemove", (e) => this.drag(e));
    document.addEventListener("mouseup", () => this.endDrag());
  }

  startDrag(e, handleId) {
    e.preventDefault();
    this.isDragging = true;
    this.activeHandle = handleId;

    if (handleId === "handle1") {
      this.handle1.classList.add("active");
    } else {
      this.handle2.classList.add("active");
    }
  }

  drag(e) {
    if (!this.isDragging || !this.activeHandle) return;

    e.preventDefault();

    const rect = this.container.getBoundingClientRect();
    const x = e.clientX - rect.left - this.centerX;
    const y = e.clientY - rect.top - this.centerY;

    let angle = (Math.atan2(y, x) * 180) / Math.PI;
    if (angle < 0) angle += 360;

    if (this.activeHandle === "handle1") {
      this.angle1 = angle;
    } else {
      this.angle2 = angle;
    }

    this.updateDisplay();
    this.notifyChange();
  }

  endDrag() {
    this.isDragging = false;
    this.activeHandle = null;
    this.handle1.classList.remove("active");
    this.handle2.classList.remove("active");
  }

  updateDisplay() {
    this.positionHandle(this.handle1, this.angle1);
    this.positionHandle(this.handle2, this.angle2);
    this.updateArcFill();
  }

  positionHandle(handle, angle) {
    const radian = (angle * Math.PI) / 180;
    const x = this.centerX + Math.cos(radian) * this.radius;
    const y = this.centerY + Math.sin(radian) * this.radius;

    handle.style.left = x + "px";
    handle.style.top = y + "px";
  }

  updateArcFill() {
    const centerX = 100;
    const centerY = 100;
    const inner_radius = 82; // Inner edge of the color wheel
    const outer_radius = 98; // outer edge

    // Convert angles to radians
    const start = (this.angle1 * Math.PI) / 180;
    const end = (this.angle2 * Math.PI) / 180;

    // Calculate start and end points on the inner circle
    const x1_r = centerX + inner_radius * Math.cos(start);
    const y1_r = centerY + inner_radius * Math.sin(start);
    const x2_r = centerX + inner_radius * Math.cos(end);
    const y2_r = centerY + inner_radius * Math.sin(end);

    // Calculate start and end points on the OUTER circle
    const x1_R = centerX + outer_radius * Math.cos(start);
    const y1_R = centerY + outer_radius * Math.sin(start);
    const x2_R = centerX + outer_radius * Math.cos(end);
    const y2_R = centerY + outer_radius * Math.sin(end);

    // Calculate the arc span
    let arcSpan = this.angle2 - this.angle1;
    // Handle wraparound. wraparound if the handles are on top of each other too
    if (arcSpan <= 0) arcSpan += 360;

    // draw full circle if the handles are on top of each other
    if (arcSpan === 360) {
      const pathData_inner = `M ${
        centerX - inner_radius
      } ${centerY} A ${inner_radius} ${inner_radius} 0 1 1 ${
        centerX + inner_radius
      } ${centerY} A ${inner_radius} ${inner_radius} 0 1 1 ${
        centerX - inner_radius
      } ${centerY}`;
      const pathData_outer = `M ${
        centerX - outer_radius
      } ${centerY} A ${outer_radius} ${outer_radius} 0 1 1 ${
        centerX + outer_radius
      } ${centerY} A ${outer_radius} ${outer_radius} 0 1 1 ${
        centerX - outer_radius
      } ${centerY}`;

      document
        .getElementById("arc-path-inner")
        .setAttribute("d", pathData_inner);
      document
        .getElementById("arc-path-outer")
        .setAttribute("d", pathData_outer);
      return;
    }

    // Determine if it's a large arc (>180 degrees)
    const largeArc = arcSpan > 180 ? 1 : 0;

    // Create the SVG inner arc path
    const pathData_inner = `M ${x1_r} ${y1_r} A ${inner_radius} ${inner_radius} 0 ${largeArc} 1 ${x2_r} ${y2_r}`;
    document.getElementById("arc-path-inner").setAttribute("d", pathData_inner);
    // outer arc path
    const pathData_outer = `M ${x1_R} ${y1_R} A ${outer_radius} ${outer_radius} 0 ${largeArc} 1 ${x2_R} ${y2_R}`;
    document.getElementById("arc-path-outer").setAttribute("d", pathData_outer);
  }

  getHueRange() {
    return {
      start: this.angle1,
      end: this.angle2,
      wrapsAround: this.angle1 > this.angle2,
    };
  }

  notifyChange() {
    if (this.onChange) {
      this.onChange(this.getHueRange());
    }
  }
}

// double sided sliders for value and chroma
class TwoHandleSlider {
  constructor(containerId, min, max, gradientCSS) {
    this.container = document.getElementById(containerId);
    this.track = this.container.querySelector(".track");
    this.range = this.container.querySelector(".range");
    this.handle1 = this.container.querySelector(".handle1");
    this.handle2 = this.container.querySelector(".handle2");

    this.min = min;
    this.max = max;
    this.value1 = min;
    this.value2 = max;

    this.col1 = "#808080";
    this.col2 = "#ff9b00";

    this.isDragging = false;
    this.activeHandle = null;
    this.onChange = null;

    this.lastHandle = null; // handle which was last touched

    this.track.style.background = gradientCSS;
    this.init();
    this.updateDisplay();
  }

  init() {
    this.handle1.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle1")
    );
    this.handle2.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle2")
    );
    document.addEventListener("mousemove", (e) => this.drag(e));
    document.addEventListener("mouseup", () => this.endDrag());
  }

  startDrag(e, handleId) {
    e.preventDefault();
    this.isDragging = true;
    this.activeHandle = handleId;
    this.container.querySelector("." + handleId).classList.add("active");
  }

  drag(e) {
    if (!this.isDragging || !this.activeHandle) return;

    const rect = this.container.getBoundingClientRect();
    let percent = (e.clientX - rect.left) / rect.width;
    percent = Math.min(Math.max(percent, 0), 1);
    const value = this.min + percent * (this.max - this.min);

    if (this.activeHandle === "handle1") {
      this.value1 = Math.min(value, this.value2); // stop overlap
    } else {
      this.value2 = Math.max(value, this.value1);
    }

    this.updateDisplay();
    if (this.onChange) this.onChange(this.getValues());
  }

  endDrag() {
    this.isDragging = false;
    this.container
      .querySelectorAll(".handle")
      .forEach((h) => h.classList.remove("active"));
  }

  updateDisplay() {
    const percent1 = (this.value1 - this.min) / (this.max - this.min);
    const percent2 = (this.value2 - this.min) / (this.max - this.min);

    this.handle1.style.left = `calc(${percent1 * 100}% - 6px)`;
    this.handle2.style.left = `calc(${percent2 * 100}% - 6px)`;

    this.range.style.left = `${percent1 * 100}%`;
    this.range.style.width = `${(percent2 - percent1) * 100}%`;
  }

  getValues() {
    return { start: this.value1, end: this.value2 };
  }

  setGradient(col1, col2) {
    this.track.style.background = `linear-gradient(to right, ${col1}, ${col2})`;
  }
}

// resize
function resize() {
  const container = document.getElementById("render-container");
  camera.aspect = container.clientWidth / container.clientHeight;
  // Recompute orthographic frustum with current "zoom" size
  const halfHeight = (camera.top - camera.bottom) / 2;
  const halfWidth = halfHeight * camera.aspect;

  camera.left = -halfWidth;
  camera.right = halfWidth;
  camera.top = halfHeight;
  camera.bottom = -halfHeight;

  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}
window.addEventListener("resize", resize);

// custom ply loader that can read hue, value, chroma properly
async function loadCustomPLY(url) {
  const response = await fetch(url);
  const text = await response.text();

  const lines = text.split("\n");
  let headerEndIndex = -1;
  let vertexCount = 0;
  let faceCount = 0;
  let vertexProperties = [];

  // Parse header
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "end_header") {
      headerEndIndex = i;
      break;
    }
    if (line.startsWith("element vertex")) {
      vertexCount = parseInt(line.split(" ")[2]);
    }
    if (line.startsWith("element face")) {
      faceCount = parseInt(line.split(" ")[2]);
    }
    if (line.startsWith("property") && !line.includes("list")) {
      const parts = line.split(" ");
      vertexProperties.push({
        type: parts[1],
        name: parts[2],
      });
    }
  }

  // Parse vertex data
  const positions = [];
  const colors = [];
  const hues = [];
  const values = [];
  const chromas = [];
  const isClipped = [];

  for (let i = headerEndIndex + 1; i < headerEndIndex + 1 + vertexCount; i++) {
    if (!lines[i]) continue; // skips empty lines etc
    const line = lines[i].trim();
    if (!line) continue;

    const values_line = line.split(" ");

    // PLY structure: x, y, z, r, g, b, hue, value, chroma, is_clipped
    positions.push(
      parseFloat(values_line[0]),
      parseFloat(values_line[1]),
      parseFloat(values_line[2])
    );

    colors.push(
      parseInt(values_line[3]) / 255,
      parseInt(values_line[4]) / 255,
      parseInt(values_line[5]) / 255
    );

    hues.push(parseFloat(values_line[6]));
    values.push(parseFloat(values_line[7]));
    chromas.push(parseFloat(values_line[8]));
    isClipped.push(parseInt(values_line[9]));
  }

  // Parse face data
  const indices = [];
  const faceStartIndex = headerEndIndex + 1 + vertexCount;

  for (
    let i = faceStartIndex;
    i < faceStartIndex + faceCount && i < lines.length;
    i++
  ) {
    const line = lines[i].trim();
    if (!line || !lines[i]) continue;

    const face_data = line.split(" ").map((v) => parseInt(v));
    const vertexCount = face_data[0];

    if (vertexCount === 3) {
      // Triangle
      indices.push(face_data[1], face_data[2], face_data[3]);
    } else if (vertexCount === 4) {
      // Quad - split into two triangles
      indices.push(
        face_data[1],
        face_data[2],
        face_data[3],
        face_data[1],
        face_data[3],
        face_data[4]
      );
    }
  }

  // Create geometry
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(positions), 3)
  );
  geometry.setAttribute(
    "color",
    new THREE.BufferAttribute(new Float32Array(colors), 3)
  );
  geometry.setAttribute(
    "hue",
    new THREE.BufferAttribute(new Float32Array(hues), 1)
  );
  geometry.setAttribute(
    "value",
    new THREE.BufferAttribute(new Float32Array(values), 1)
  );
  geometry.setAttribute(
    "chroma",
    new THREE.BufferAttribute(new Float32Array(chromas), 1)
  );
  geometry.setAttribute(
    "isClipped",
    new THREE.BufferAttribute(new Float32Array(isClipped), 1)
  );

  if (indices.length > 0) {
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
  }

  return geometry;
}

// Generalized mesh loader
function loadMeshes() {
  const meshConfigs = [
    {
      file: "./munsell_mesh.ply",
      name: "shell",
      type: "mesh",
      materials: {
        mesh: async () => await slicer.getMaterial("mesh"),
      },
    },
    // interpolated point cloud
    // tbh this should never get shown
    {
      file: "./munsell_pointcloud_interpolated.ply",
      name: "pointcloud_interpolated",
      type: "points",
      materials: {
        points: async () => await slicer.getMaterial("points"),
      },
    },
    // raw real.dat data points
    {
      file: "./munsell_pointcloud_original.ply",
      name: "pointcloud_original",
      type: "points",
      materials: {
        points: async () => await slicer.getMaterial("points"),
      },
    },
  ];

  let loadedCount = 0;
  const totalMeshes = meshConfigs.length;

  meshConfigs.forEach((config) => {
    loadCustomPLY(config.file).then(async (geometry) => {
      // Create materials
      const materials = {};
      for (const [key, materialFactory] of Object.entries(config.materials)) {
        materials[key] = await materialFactory();
      }

      // Create Three.js object
      let threejsObject;
      if (config.type === "mesh") {
        threejsObject = new THREE.Mesh(geometry, materials.mesh);
      } else if (config.type === "points") {
        threejsObject = new THREE.Points(geometry, materials.points);
      }

      // Store mesh data
      const meshObj = {
        geometry,
        materials,
        mesh: threejsObject,
        config,
      };

      scene.add(threejsObject);
      meshes[config.name] = meshObj;
      loadedCount++;

      // when we load the shell mesh, center the camera on it
      if (config.name === "shell") {
        centerCamera(threejsObject);
      }

      // After all meshes are loaded, set the default scene
      if (loadedCount === totalMeshes) {
        // initStencil(meshes["shell"]);
        switchScene("default");
      }
    });
  });
}

// fit camera to be aligned with/look at mesh
// offset = leftwards offset of the mesh from the centre of the screen
function centerCamera(object, scale = 1, offset = 0.167) {
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  // move controls target to mesh center
  controls.target.copy(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  const aspect =
    renderer.domElement.clientWidth / renderer.domElement.clientHeight;

  // offset camera a bit so that there is space at the right for ui controls
  const offsetX = offset * maxDim;

  camera.left = (-maxDim * aspect * 0.5) / scale + offsetX;
  camera.right = (maxDim * aspect * 0.5) / scale + offsetX;
  camera.top = (maxDim * 0.5) / scale;
  camera.bottom = (-maxDim * 0.5) / scale;
  camera.near = -maxDim * 2;
  camera.far = maxDim * 2;
  camera.updateProjectionMatrix();

  camera.position.set(center.x, center.y, center.z + maxDim / scale);

  camera.lookAt(center);

  // keep OrbitControls centred around the mesh
  controls.target.copy(center);
  controls.update();
}

// animate
function animate() {
  requestAnimationFrame(animate);
  if (!isPaused) {
    for (const m in meshes) {
      meshes[m].mesh.rotation.y += 0.01;
    }
  }

  // clear color, depth, & stencil buffers
  renderer.clear(true, true, true);
  renderer.render(scene, camera);

  controls.update();
}

async function load3DTexture(filename, size) {
  const response = await fetch(filename);
  if (!response.ok) {
    throw new Error(`Failed to load texture: ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const data = new Uint8Array(arrayBuffer);

  const texture = new THREE.Data3DTexture(data, size, size, size);
  texture.format = THREE.RGBAFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;

  return texture;
}

function main() {
  initScene();
  slicer = new Slicer();
  loadMeshes();

  initUI();
  resize();
  animate();
}

main();
