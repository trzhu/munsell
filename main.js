import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';
import { PLYLoader } from 'PLYLoader';

let isPaused = false;
let mesh = null; // reference to loaded mesh
let litMaterial, unlitMaterial;

// scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x7f7f7f);

// camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 10);

// renderer
const renderer = new THREE.WebGLRenderer({ antialias: true });
document.getElementById('render-container').appendChild(renderer.domElement);

// controls
const controls = new OrbitControls(camera, renderer.domElement);

// resize
function resize() {
  const container = document.getElementById('render-container');
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}
window.addEventListener('resize', resize);
resize();

// lights
const light = new THREE.DirectionalLight(0xffffff, 5);
light.position.set(5, 5, 5);
scene.add(light);
scene.add(new THREE.AmbientLight(0x404040));

// load PLY
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

  mesh = new THREE.Mesh(geometry, litMaterial);
  scene.add(mesh);

  geometry.computeBoundingSphere();
  if (geometry.boundingSphere) {
    camera.position.z = geometry.boundingSphere.radius * 3;
  }
});

// buttons
const pauseButton = document.getElementById('toggle-rotation');
pauseButton.addEventListener('click', () => {
  isPaused = !isPaused;
  pauseButton.textContent = isPaused ? 'Play Rotation' : 'Pause Rotation';
});

// add lighting toggle button
const toggleLightButton = document.createElement('button');
toggleLightButton.textContent = 'Show Exact Color';
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

// animate
function animate() {
  requestAnimationFrame(animate);
  if (!isPaused && mesh) {
    mesh.rotation.y += 0.01;
  }
  renderer.render(scene, camera);
}
animate();
