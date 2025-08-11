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

  // Initialize circular hue slider
  const circularHueSlider = new CircularHueSlider("hue-slider");
  circularHueSlider.onChange = (range) => {
    console.log("Hue range:", range);
    // Connect to your slicer here
  };

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

    this.clippingPlanes = [
      this.horizontal,
      this.radial_lower,
      this.radial_upper,
    ];
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
    const baseAngle = (this.baseHueAngle * Math.PI) / 180 + meshRotationY;
    const halfSpan = (this.hueSpan * Math.PI) / 180 / 2;

    // Lower bound of pie slice
    this.radial_lower.normal.set(
      Math.cos(baseAngle - halfSpan),
      0,
      Math.sin(baseAngle - halfSpan)
    );

    // Upper bound of pie slice (note the flipped normal for intersection)
    this.radial_upper.normal.set(
      -Math.cos(baseAngle + halfSpan),
      0,
      -Math.sin(baseAngle + halfSpan)
    );
  }
}

class CircularHueSlider {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.handle1 = document.getElementById("handle1");
    this.handle2 = document.getElementById("handle2");
    this.arcFill = document.getElementById("arc-fill");

    this.centerX = 100;
    this.centerY = 100;
    this.radius = 90;

    this.angle1 = 0;
    this.angle2 = 60;

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
    if (arcSpan < 0) arcSpan += 360; // Handle wraparound

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

// making this a helper bc later hue angle might not be 0-360
function hueToRadians(hueAngle) {
  return (hueAngle * Math.PI) / 180;
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
