import * as THREE from 'three';
import { OrbitControls } from 'OrbitControls';

let isPaused = false;

async function loadShader(url) {
    const response = await fetch(url);
    return await response.text();
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x7f7f7f);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
document.getElementById('render-container').appendChild(renderer.domElement);

// buttons 
const pauseButton = document.getElementById('Pause');
pauseButton.addEventListener('click', () => {
    isPaused = !isPaused;
    pauseButton.textContent = isPaused ? 'Play' : 'Pause';
});

function resize() {
  const container = document.getElementById('render-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}
window.addEventListener('resize', resize);
resize();

// load shaders (cooked rn)
const vertexShader = await loadShader('./shaders/hsvsphere_frag.glsl');
const fragmentShader = await loadShader('./shaders/hsvsphere_vert.glsl');

// Controls
const controls = new OrbitControls(camera, renderer.domElement);

// placeholder green cube
const geometry = new THREE.BoxGeometry(2, 2, 2);
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });

// placeholder cube
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// it rotat
function animate() {
  requestAnimationFrame(animate);
  if (!isPaused) {
    cube.rotation.y += 0.005;
  }
  renderer.render(scene, camera);
}

animate();

