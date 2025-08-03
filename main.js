import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';
import { PLYLoader } from 'PLYLoader';

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
  const container = document.getElementById('render-container');
  const aspect = container.clientWidth / container.clientHeight;
  // orthographic camera setup
  // parameters dont matter bc we're gonna centre it to our mesh later anyways
  camera = new THREE.OrthographicCamera(
    -1 * aspect, 1 * aspect, 1, -1, 
    0.1, 5000
  );

  // renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  document.getElementById('render-container').appendChild(renderer.domElement);

  // controls
  controls = new OrbitControls(camera, renderer.domElement);

  // init lights
  const light = new THREE.DirectionalLight(0xffffff, 3.5);
  light.position.set(5, 5, 5);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xeeeeee));
}

function initUI() {
  // BUTTONS
  // play/pause button
  const pauseButton = document.getElementById('toggle-rotation');
  pauseButton.addEventListener('click', () => {
    isPaused = !isPaused;
    pauseButton.textContent = isPaused ? 'Play Rotation' : 'Pause Rotation';
  });

  // lighting toggle button
  const toggleLightButton = document.createElement('button');
  toggleLightButton.textContent = 'Turn on Lighting';
  toggleLightButton.style.width = '100%';
  document.getElementById('floating-ui').appendChild(toggleLightButton);

  toggleLightButton.addEventListener('click', () => {
    if (mesh) {
      if (mesh.material === litMaterial) {
        mesh.material = unlitMaterial;
        toggleLightButton.textContent = 'Turn on Lighting';
      } else {
        mesh.material = litMaterial;
        toggleLightButton.textContent = 'Show Exact Color';
      }
    }
  });
}

class Slicer {
  constructor() {
    this.horizontalPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
    this.radialPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
    this.clippingPlanes = [this.horizontalPlane, this.radialPlane];
  }

  applyToMaterial(material) {
    material.clippingPlanes = this.clippingPlanes;
    material.clipIntersection = true;
  }

  setHorizontal(offset) {
    this.horizontalPlane.constant = offset;
  }

  setRadial(angleDeg) {
    const theta = angleDeg * Math.PI / 180;
    this.radialPlane.normal.set(Math.cos(theta), 0, Math.sin(theta));
  }
}

// resize
function resize() {
  const container = document.getElementById('render-container');
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}
window.addEventListener('resize', resize);

// load mesh from PLY file
function loadMesh() {
  const loader = new PLYLoader();
  loader.load('./munsell_mesh.ply', (geometry) => {
    geometry.computeVertexNormals();

    litMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      flatShading: false,
      vertexColors: geometry.hasAttribute('color')
    });

    unlitMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      vertexColors: geometry.hasAttribute('color')
    });

    mesh = new THREE.Mesh(geometry, unlitMaterial);
    scene.add(mesh);

    centerCamera(mesh);
  }); 
}

// fit camera to be aligned with/look at mesh
function centerCamera(object, offset = 1) {
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  // move controls target to mesh center
  controls.target.copy(center);

  const maxDim = Math.max(size.x, size.y, size.z);
  const aspect = renderer.domElement.clientWidth / renderer.domElement.clientHeight;

  camera.left   = -maxDim * aspect * 0.5 * offset;
  camera.right  =  maxDim * aspect * 0.5 * offset;
  camera.top    =  maxDim * 0.5 * offset;
  camera.bottom = -maxDim * 0.5 * offset;
  camera.near = 0.001;
  camera.far = 1000;
  camera.updateProjectionMatrix();

  camera.position.set(center.x, center.y, center.z + maxDim * offset);

  camera.lookAt(center);
}


// animate
function animate() {
  requestAnimationFrame(animate);
  if (!isPaused && mesh) {
    mesh.rotation.y += 0.01;
  }
  renderer.render(scene, camera);
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