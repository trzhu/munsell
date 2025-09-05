import * as THREE from "three";
import { OrbitControls } from "OrbitControls";
import { Slicer } from "./slicer.js";
import { CircularSlider, TwoHandleSlider } from "./ui.js";

// globals
let scene, camera, renderer, controls;
let slicer;
let isPaused = false;

// dictionary of meshes
// keys: "shell", "pointcloud", "pointcloud_original, cutSurfaces"
// todo: maybe just store the meshes instead of the whole meshobj. i dont bother with any of the other fields anyways
const meshes = {};

const sceneConfigs = {
  default: {
    name: "Volume",
    // visible: ["shell"],
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
    visible: ["cutSurfaces"],
    // visible: ["shell", "cutSurfaces"],
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
    updateCutSurfaces();
  };
  circularHueSlider.onChange(circularHueSlider.getHueRange());

  // two-handle linear value slider
  const valueSlider = new TwoHandleSlider("value-slider", 0, 10);
  valueSlider.onChange = (range) => {
    slicer.setValueRange(range.start, range.end);
    updateCutSurfaces();
  };
  valueSlider.onChange(valueSlider.getValues());

  // two-handle linear chroma slider
  const chromaSlider = new TwoHandleSlider("chroma-slider", 0, 38);
  chromaSlider.onChange = (range) => {
    slicer.setChromaRange(range.start, range.end);
    updateCutSurfaces();
  };
  // idk initalize it to some random colors
  chromaSlider.setGradient("#808080", "#ff0000");
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
// disposes of old surfaces, gets new cut surfaces from slicer, and adds it to scene
async function updateCutSurfaces() {
  // new surfaces from slicer
  const meshObj = await slicer.createCutSurfaces();

  if (meshObj) {
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

    // copy y rotation from shell mesh
    if (meshes["shell"]?.mesh) {
      meshObj.mesh.rotation.copy(meshes["shell"].mesh.rotation);
    }

    // add to scene and meshes dictionary
    scene.add(meshObj.mesh);
    meshes["cutSurfaces"] = meshObj;
  }
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
