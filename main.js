import * as THREE from "three";
import { OrbitControls } from "OrbitControls";
// import { PLYLoader } from "PLYLoader";

// globals
let scene, camera, renderer, controls;
let slicer;
let isPaused = false;

const meshes = {}; // dictionary of meshes
// keys: "shell", "pointcloud", "pointcloud_original"

let litMaterial, unlitMaterial;

// Scene configurations
const sceneConfigs = {
  default: {
    name: "Volume",
    visible: ["shell"],
    hidden: ["pointcloud_interpolated", "pointcloud_original"],
  },
  pointCloud: {
    name: "Points",
    visible: ["pointcloud_original"],
    hidden: ["shell", "pointcloud_interpolated"],
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

  // renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.localClippingEnabled = true;
  document.getElementById("render-container").appendChild(renderer.domElement);

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
    if (meshes["shell"]) {
      const shellMesh = meshes["shell"].mesh;
      if (shellMesh.material === litMaterial) {
        shellMesh.material = unlitMaterial;
        toggleLightButton.textContent = "Turn on Lighting";
      } else {
        shellMesh.material = litMaterial;
        toggleLightButton.textContent = "Show Exact Color";
      }
    }
  });

  // Initialize circular hue slider
  const circularHueSlider = new CircularSlider("hue-slider");
  circularHueSlider.onChange = (range) => {
    slicer.setHueRange(range.start, range.end);
    // todo: change chroma slider's colours on change as well
  };
  circularHueSlider.onChange(circularHueSlider.getHueRange());

  // two-handle linear value slider
  const valueSlider = new TwoHandleSlider("value-slider", 0, 10);
  valueSlider.onChange = (range) => {
    slicer.setValueRange(range.start, range.end);
    // todo: change chroma slider's colours on change as well
  };
  valueSlider.onChange(valueSlider.getValues());

  // two-handle linear chroma slider
  const chromaSlider = new TwoHandleSlider("chroma-slider", 0, 38);
  chromaSlider.onChange = (range) => {
    slicer.setChromaRange(range.start, range.end);
  };
  chromaSlider.onChange(chromaSlider.getValues());

  const sceneSelect = document.getElementById("sceneSelect");

  sceneSelect.addEventListener("change", (event) => {
    const selectedScene = event.target.value;
    switchScene(selectedScene);
  });
}

// Scene switching function
function switchScene(sceneKey) {
  const config = sceneConfigs[sceneKey];

  if (!config) return;

  // Hide all meshes first
  Object.keys(meshes).forEach((meshName) => {
    if (meshes[meshName] && meshes[meshName].mesh) {
      meshes[meshName].mesh.visible = false;
    }
  });

  // Show only the meshes specified in the scene config
  config.visible.forEach((meshName) => {
    if (meshes[meshName] && meshes[meshName].mesh) {
      meshes[meshName].mesh.visible = true;
    }
  });

  const toggleLightButton = document.getElementById("toggle-light");
  if (sceneKey === "default") {
    toggleLightButton.style.display = "block";
  } else {
    toggleLightButton.style.display = "none";
  }
}

class Slicer {
  constructor() {
    this.uniforms = {
      hueMin: { value: 0.0 },
      hueMax: { value: 360.0 },
      chromaMin: { value: 0.0 },
      chromaMax: { value: 38.0 },
      valueMin: { value: 0.0 },
      valueMax: { value: 10.0 },
      uSize: {value: 10.0}
    };

    this.shadersPromise = this.loadShaders();
  }

  async loadShaders() {
    const [meshVertex, meshFragment, pointsVertex, pointsFragment] =
      await Promise.all([
        fetch("./shaders/mesh_vertex.glsl").then((r) => r.text()),
        fetch("./shaders/mesh_fragment.glsl").then((r) => r.text()),
        fetch("./shaders/points_vertex.glsl").then((r) => r.text()),
        fetch("./shaders/points_fragment.glsl").then((r) => r.text()),
      ]);

    return { meshVertex, meshFragment, pointsVertex, pointsFragment };
  }

  async getMaterial(type = "points") {
    const shaders = await this.shadersPromise;

    let vertexShader, fragmentShader;
    if (type === "points") {
      vertexShader = shaders.pointsVertex;
      fragmentShader = shaders.pointsFragment;
    } else if (type === "mesh") {
      vertexShader = shaders.meshVertex;
      fragmentShader = shaders.meshFragment;
    } else {
      throw new Error(`Unsupported type: ${type}`);
    }

    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: this.uniforms,
      transparent: true,
    });
  }

  setHueRange(min, max) {
    this.uniforms.hueMin.value = min;
    this.uniforms.hueMax.value = max;
  }

  setChromaRange(min, max) {
    this.uniforms.chromaMin.value = min;
    this.uniforms.chromaMax.value = max;
  }

  setValueRange(min, max) {
    this.uniforms.valueMin.value = min;
    this.uniforms.valueMax.value = max;
  }
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
      const pathData_inner = `M ${centerX - inner_radius} ${centerY} A ${inner_radius} ${inner_radius} 0 1 1 ${centerX + inner_radius} ${centerY} A ${inner_radius} ${inner_radius} 0 1 1 ${centerX - inner_radius} ${centerY}`;
      const pathData_outer = `M ${centerX - outer_radius} ${centerY} A ${outer_radius} ${outer_radius} 0 1 1 ${centerX + outer_radius} ${centerY} A ${outer_radius} ${outer_radius} 0 1 1 ${centerX - outer_radius} ${centerY}`;

      document.getElementById("arc-path-inner").setAttribute("d", pathData_inner);
      document.getElementById("arc-path-outer").setAttribute("d", pathData_outer);
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

    this.isDragging = false;
    this.activeHandle = null;
    this.onChange = null;

    this.track.style.background = gradientCSS;
    this.init();
    this.updateDisplay();
  }

  init() {
    this.handle1.addEventListener("mousedown", e => this.startDrag(e, "handle1"));
    this.handle2.addEventListener("mousedown", e => this.startDrag(e, "handle2"));
    document.addEventListener("mousemove", e => this.drag(e));
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
    this.container.querySelectorAll(".handle").forEach(h => h.classList.remove("active"));
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

  setGradient(colors) {
    if (Array.isArray(colors) && colors.length > 1) {
      this.track.style.background = `linear-gradient(to right, ${colors.join(", ")})`;
    }
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
  
  const lines = text.split('\n');
  let headerEndIndex = -1;
  let vertexCount = 0;
  let faceCount = 0;
  let vertexProperties = [];
  
  // Parse header
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === 'end_header') {
      headerEndIndex = i;
      break;
    }
    if (line.startsWith('element vertex')) {
      vertexCount = parseInt(line.split(' ')[2]);
    }
    if (line.startsWith('element face')) {
      faceCount = parseInt(line.split(' ')[2]);
    }
    if (line.startsWith('property') && !line.includes('list')) {
      const parts = line.split(' ');
      vertexProperties.push({
        type: parts[1],
        name: parts[2]
      });
    }
  }
  
  // console.log(`Loading PLY: ${vertexCount} vertices, ${faceCount} faces`);
  // console.log('Vertex properties:', vertexProperties.map(p => p.name));
  
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
    
    const values_line = line.split(' ');
    
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
  
  for (let i = faceStartIndex; i < faceStartIndex + faceCount && i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line || !lines[i]) continue;
    
    const face_data = line.split(' ').map(v => parseInt(v));
    const vertexCount = face_data[0];
    
    if (vertexCount === 3) {
      // Triangle
      indices.push(face_data[1], face_data[2], face_data[3]);
    } else if (vertexCount === 4) {
      // Quad - split into two triangles
      indices.push(
        face_data[1], face_data[2], face_data[3],
        face_data[1], face_data[3], face_data[4]
      );
    }
  }
  
  // Create geometry
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
  geometry.setAttribute('hue', new THREE.BufferAttribute(new Float32Array(hues), 1));
  geometry.setAttribute('value', new THREE.BufferAttribute(new Float32Array(values), 1));
  geometry.setAttribute('chroma', new THREE.BufferAttribute(new Float32Array(chromas), 1));
  geometry.setAttribute('isClipped', new THREE.BufferAttribute(new Float32Array(isClipped), 1));
  
  if (indices.length > 0) {
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
  }
  
  // console.log(`Loaded: ${positions.length/3} vertices, ${indices.length/3} faces`);
  
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
        lit: async () => await slicer.getMaterial("mesh"),
        unlit: () =>
          new THREE.MeshBasicMaterial({
            vertexColors: true,
          }),
      },
      postProcess: (geometry, meshObj) => {
        geometry.computeVertexNormals();

        // TODO: APPLY SLICING to mesh

        litMaterial = meshObj.materials.lit;
        unlitMaterial = meshObj.materials.unlit;
      },
    },
    // interpolated point cloud
    // tbh this should never get shown
    {
      file: "./munsell_pointcloud.ply",
      name: "pointcloud_interpolated",
      type: "points",
      materials: {
        points: async () => await slicer.getMaterial("points"),
      },
      postProcess: (geometry, meshObj) => {
        // Apply custom shader for slicing
        // slicer.applyToPointMaterial(meshObj.materials.points);
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
      postProcess: (geometry, meshObj) => {
        // no post-process for original point cloud
      },
    },
  ];

  // const loader = new PLYLoader();
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
        threejsObject = new THREE.Mesh(
          geometry,
          materials.unlit || materials.lit
        );
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

      // Run post-processing
      if (config.postProcess) {
        config.postProcess(geometry, meshObj);
      }

      scene.add(threejsObject);
      meshes[config.name] = meshObj;
      loadedCount++;

      // when we load the shell mesh, center the camera on it
      if (config.name === "shell") {
        centerCamera(threejsObject);
      }

      // After all meshes are loaded, set the default scene
      if (loadedCount === totalMeshes) {
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

  renderer.render(scene, camera);

  controls.update();
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
