import * as THREE from "three";
import { OrbitControls } from "OrbitControls";
import { PLYLoader } from "PLYLoader";

// globals
let scene, camera, renderer, controls;
let slicer;
let isPaused = false;

let mesh = null; // reference to loaded mesh
let litMaterial, unlitMaterial;

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
  const toggleLightButton = document.createElement("button");
  toggleLightButton.textContent = "Turn on Lighting";
  toggleLightButton.style.width = "100%";
  document.getElementById("floating-ui").appendChild(toggleLightButton);

  toggleLightButton.addEventListener("click", () => {
    if (mesh) {
      if (mesh.material === litMaterial) {
        mesh.material = unlitMaterial;
        toggleLightButton.textContent = "Turn on Lighting";
      } else {
        mesh.material = litMaterial;
        toggleLightButton.textContent = "Show Exact Color";
      }
    }
  });

  // Hue slider
  document.querySelector("#hue-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    slicer.setHue(val);
  });

  // Chroma slider
  document.querySelector("#chroma-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    slicer.setChroma(val);
    // TODO: chroma not hooked up yet
  });

  // Value slider
  document.querySelector("#value-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    slicer.setValue(val);
  });
}

// TODO: look into stencils to see if that can help me colour in the clipped parts
// either that or I'll create faces for every fin
class Slicer {
  constructor() {
    // horizontal plane - cuts along value axis
    this.horizontal = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

    // radial planes (cut around hue axis)
    this.radial_lower = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
    this.radial_upper = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);

    // lower hue angle and hue span angle
    this.baseHueAngle = 0;
    this.hueSpan = 170;

    this.clippingPlanes = [this.horizontal, this.radial_lower, this.radial_upper];
  }

  applyToMaterial(material) {
    material.clippingPlanes = this.clippingPlanes;
    material.clipIntersection = false;
    material.needsUpdate = true;
  }

  // horizontal (Value)
  setValue(offset) {
    // mesh height is 30
    // need negative offset for it to work
    this.horizontal.constant = -offset;
  }

  // radial (Hue)
  setHue(angleDeg) {
    this.baseHueAngle = angleDeg;
    const theta = (angleDeg * Math.PI) / 180;
    this.radial_lower.normal.set(Math.cos(theta), 0, Math.sin(theta));
    this.radial_lower.constant = 0;
  }

  setHueSpan(spanDegrees) {
    this.hueSpan = spanDegrees;
  }


  // chroma (radius cutoff) (needs shader OR bounding logic)
  setChroma(maxRadius) {
    this.maxRadius = maxRadius;
    // TODO LMAO IDK WHATS GOING ON HERE
  }

  updateRadialPlaneRotation(meshRotationY) {
  const baseAngle = (this.baseHueAngle * Math.PI / 180) + meshRotationY;
  const halfSpan = (this.hueSpan * Math.PI / 180) / 2;
  
  // Lower bound of pie slice
  this.radial_lower.normal.set(Math.cos(baseAngle - halfSpan), 0, Math.sin(baseAngle - halfSpan));
  
  // Upper bound of pie slice (note the flipped normal for intersection)
  this.radial_upper.normal.set(-Math.cos(baseAngle + halfSpan), 0, -Math.sin(baseAngle + halfSpan));
  }
}

// making this a helper bc later hue angle might not be 0-360
function hueToRadians(hueAngle) {
  return hueAngle * Math.PI / 180
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

// load mesh from PLY file
// TODO: this one loads the solid mesh 
// later might load point cloud as well
function loadMesh() {
  const loader = new PLYLoader();
  loader.load("./munsell_mesh.ply", (geometry) => {
    geometry.computeVertexNormals();

    litMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true,
      clipIntersection: true,
    });

    unlitMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      clipIntersection: true,
    });

    slicer.applyToMaterial(litMaterial);
    slicer.applyToMaterial(unlitMaterial);

    mesh = new THREE.Mesh(geometry, unlitMaterial);
    scene.add(mesh);

    centerCamera(mesh);
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
  // offset camera a bit so that
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
  if (!isPaused && mesh) {
    mesh.rotation.y += 0.01;
  }

  // Update radial plane to rotate with mesh
  if (slicer) {
    slicer.updateRadialPlaneRotation(-mesh.rotation.y);
  }

  renderer.render(scene, camera);

  controls.update();
}

function main() {
  initScene();
  loadMesh();
  slicer = new Slicer();
  initUI();
  resize();
  animate();
}

main();
