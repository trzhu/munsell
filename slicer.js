import * as THREE from "three";

// Y scaling factor used for mesh
const Y_SCALE = 3;

class Slicer {
  constructor() {
    // stuff that gets passed into shader
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
    this.loadTextures();

    // load max chroma dictionary from jsons
    this.maxChromaPromise = loadMaxChromaDict();
    this.maxChroma = null;
    this.maxChromaPromise.then((data) => {
      this.maxChroma = data;
    });

    // pointer for future cut surface material
    this.cutSurfaceMaterial = null;

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

    // return cached cutSurface material if it exists
    if (type === "cutSurface" && this.cutSurfaceMaterial) {
      return this.cutSurfaceMaterial;
    }

    let vertexShader, fragmentShader, side;
    if (type === "points") {
      vertexShader = shaders.pointsVertex;
      fragmentShader = shaders.pointsFragment;
    } else if (type === "mesh") {
      vertexShader = shaders.meshVertex;
      fragmentShader = shaders.meshFragment;
    } else if (type === "cutSurface") {
      vertexShader = shaders.sliceVertex;
      fragmentShader = shaders.sliceFragment;
    } else {
      throw new Error(`Unsupported type: ${type}`);
    }

    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: this.uniforms,
      transparent: true,
    });

    // cache cut surface material bc it needs to be retrieved multiple times
    if (type === "cutSurface") {
      this.cutSurfaceMaterial = material;
      // debug
      // material.wireframe = true;
    }

    return material;
  }

  setHueRange(min, max) {
    this.uniforms.hueMin.value = (min * Math.PI) / 180;
    this.uniforms.hueMax.value = (max * Math.PI) / 180;
    this.createCutSurfaces();
  }

  setChromaRange(min, max) {
    this.uniforms.chromaMin.value = min;
    this.uniforms.chromaMax.value = max;
    this.createCutSurfaces();
  }

  setValueRange(min, max) {
    this.uniforms.valueMin.value = min;
    this.uniforms.valueMax.value = max;
    this.createCutSurfaces();
  }

  toggleLighting() {
    this.uniforms.useLighting.value = 1 - this.uniforms.useLighting.value;
  }

  toggleRGB() {
    this.uniforms.showOutsideRGB.value = 1 - this.uniforms.showOutsideRGB.value;
  }

  // creates cut surface geometry with current min/max h,v,c values
  async createCutSurfaces() {
    if (!this.maxChroma) {
      // console.log("maxChroma still loading");
      return;
    }

    // clear references
    Object.keys(this.meshRefs).forEach((key) => {
      this.meshRefs[key] = null;
    });

    const group = new THREE.Group();

    const materialPromise = this.getMaterial("cutSurface");
    return materialPromise.then((material) => {
      // generates valid values to loop over - minH, (grid points in between), ... maxH
      const hueLoop = this.hueLoop(
        this.uniforms.hueMin.value,
        this.uniforms.hueMax.value
      );

      // create geometry for each cut surface
      const surfaces = {
        hueMinPlane: this.cutSurface("hue", false),
        hueMaxPlane: this.cutSurface("hue", true),
        valueMinPlane: this.cutSurface("value", false, hueLoop),
        valueMaxPlane: this.cutSurface("value", true, hueLoop),
        chromaMinCyl: this.cutSurface("chroma", false, hueLoop),
        chromaMaxCyl: this.cutSurface("chroma", true, hueLoop),
      };

      // persist all surfaces to meshRefs
      Object.entries(surfaces).forEach(([key, geometry]) => {
        this.meshRefs[key] = new THREE.Mesh(geometry, material);
      });

      // add all meshes to a group
      Object.values(this.meshRefs).forEach((mesh) => {
        if (mesh) {
          group.add(mesh);
        }
      });

      // define the cut surface mesh to be this whole group
      const meshObj = {
        materials: material,
        mesh: group,
        config: "cutSurface",
      };

      return meshObj;
    });
  }

  cutSurface(surfaceType, isMax, hueSequence = null) {
    const minH = this.uniforms.hueMin.value * 180 / Math.PI, maxH = this.uniforms.hueMax.value * 180 / Math.PI;
    const minV = this.uniforms.valueMin.value, maxV = this.uniforms.valueMax.value;
    const minC = this.uniforms.chromaMin.value, maxC = this.uniforms.chromaMax.value;
    
    // surface type implies that is the fixed coordinate
    let fixed;
    if (surfaceType === "hue") {
      fixed = isMax ? maxH : minH;
    } else if (surfaceType === "value") {
      fixed = isMax ? maxV : minV;
    } else if (surfaceType === "chroma") {
      fixed = isMax ? maxC : minC;
    }

    // set the values we're gonna loop over - value for hue planes, hue for the other 2
    let loop;
    if (surfaceType === "hue") {
      loop = [minV];
      for (let v = Math.floor(minV + 1); v < maxV; v += 1) {
        loop.push(v);
      }
      loop.push(maxV);
    } else {
      loop = hueSequence;
    }

    // smaller numbers = lower or inner edge loop
    // bigger numbers = upper or outer edge loop
    const vertices1 = [];
    const vertices2 = [];

    let windingOrder;
    // for each point in the varying coordinate,
    for (const val of loop) {
      if (surfaceType === "hue") {
        // hue=fixed, value=varying, chroma=clamped
        const h = fixed;
        const v = val;
        const maxChroma = this.getMaxChroma(h, v);

        vertices1.push(this.HVC_to_XYZ(h, v, minC));
        vertices2.push(this.HVC_to_XYZ(h, v, clamp(maxChroma, minC, maxC))); // outer edge clamped

        windingOrder = isMax;
      } else if (surfaceType === "value") {
        // hue=varying, value=fixed, chroma=clamped
        const h = val;
        const v = fixed;
        const maxChroma = this.getMaxChroma(h, v);

        vertices1.push(this.HVC_to_XYZ(h, v, minC));
        vertices2.push(this.HVC_to_XYZ(h, v, clamp(maxChroma, minC, maxC))); // outer edge clamped

        windingOrder = !isMax;
      } else if (surfaceType === "chroma") {
        // hue=varying, chroma=fixed, value=clamped
        const h = val;
        const c = fixed;

        // find valid value range for this hue at this chroma
        const { validMinV, validMaxV } = this.valueRange(h, c, minV, maxV);
        const clampedMinV = clamp(minV, validMinV, validMaxV);
        const clampedMaxV = clamp(maxV, validMinV, validMaxV);

        // if one of the clamped values was null, push null so that createGeometry knows to skip the vertex
        vertices1.push(clampedMinV === null ? null : this.HVC_to_XYZ(h, clampedMinV, c));
        vertices2.push(clampedMaxV === null ? null : this.HVC_to_XYZ(h, clampedMaxV, c));

        windingOrder = isMax;
      }
    }

    return createGeometry(vertices1, vertices2, windingOrder);
  }

  // finds minimum and maximum value that is inside the volume at this chroma
  // TODO: PROBLEM IS THIS ONLY CHECKS INTEGERS SO THERE ARE GAPS
  // TODO: WHY DOES IT ACT WEIRD FOR SOME HUE VALUES
  valueRange(h, c, minV, maxV) {
    let left = Math.floor(minV), right = Math.ceil(maxV);
    let validMinV = null, validMaxV = null;

    // two pointer approach
    while (left <= right && (validMinV === null || validMaxV === null)) {
      // check left pointer if we haven't found validMinV yet
      if (validMinV === null) {
        const maxChromaLeft = this.getMaxChroma(h, left);
        if (c <= maxChromaLeft) {
          validMinV = left;
        } else {
          left++;
        }
      }
      // check right pointer if we haven't found validMaxV yet
      if (validMaxV === null && left <= right) {
        const maxChromaRight = this.getMaxChroma(h, right);
        if (c <= maxChromaRight) {
          validMaxV = right;
        } else {
          right--;
        }
      }
    }

    // if no valid min or max was found, intentionally return null
    return {
      validMinV,
      validMaxV,
    };
  }

  hueLoop(minH, maxH) {
    // handle hue wraparound
    minH = (minH * 180) / Math.PI;
    maxH = (maxH * 180) / Math.PI;
    const hueSpan = maxH > minH ? maxH - minH : 360 - minH + maxH;
    const numSteps = Math.floor(hueSpan / 9);
    const startH = Math.floor(minH / 9) * 9 + 9;

    const sequence = [minH];
    // for every h from minH to maxH in increments of 9 (the grid points)
    for (let i = 1; i < numSteps; i++) {
      sequence.push((startH + i * 9) % 360);
    }
    sequence.push(maxH);

    return sequence;
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

  // max possible chroma at this hue and value combination
  // returns direct value form munsell data points if it exists, lerps otherwise
  getMaxChroma(hue, value) {
    // edge cases white and black
    if (value <= 0 || value >= 10) {
      return 0;
    }

    // if already on the grid
    if (this.maxChroma[hue] && this.maxChroma[hue][value]) {
      return this.maxChroma[hue][value];
    }

    const hStep = 9;

    let lowHue = Math.floor(hue / hStep) * hStep;
    let highHue = lowHue + hStep;

    // handle hue wraparound at 0/360 boundary
    lowHue = lowHue % 360;
    highHue = highHue % 360;
    if (lowHue === 0) {
      lowHue = 360;
    } else if (highHue === 0) {
      highHue = 360;
    }

    // Find surrounding value levels (integers)
    const lowValue = Math.floor(value);
    const highValue = Math.min(10, lowValue + 1);

    // Get the 4 corner values for bilinear interpolation
    const c00 = this.maxChroma[lowHue][lowValue] || 0; // bottom-left
    const c10 = this.maxChroma[highHue][lowValue] || 0; // bottom-right
    const c01 = this.maxChroma[lowHue][highValue] || 0; // top-left
    const c11 = this.maxChroma[highHue][highValue] || 0; // top-right

    // Calculate interpolation weights
    let hueWeight;
    if (lowHue === 360 && highHue === 9) {
      // Special case for wraparound
      if (hue <= 180) {
        // targetHue is closer to 9 than 360
        hueWeight = (hue + 360 - 360) / (9 + 360 - 360);
      } else {
        // targetHue is closer to 360
        hueWeight = (hue - 360) / (9 + 360 - 360);
      }
    } else {
      hueWeight = (hue - lowHue) / (highHue - lowHue);
    }

    const valueWeight = (value - lowValue) / (highValue - lowValue);

    // bilinear interpolation
    const c0 = c00 * (1 - hueWeight) + c10 * hueWeight;
    const c1 = c01 * (1 - hueWeight) + c11 * hueWeight;
    const result = c0 * (1 - valueWeight) + c1 * valueWeight;

    // console.log("hue, value, result: ", hue, value, result);

    return result;
  }
}

function clamp(num, min, max) {
  if (num === null || min === null || max === null) {
    return null;
  }
  return Math.min(Math.max(num, min), max);
}

async function loadMaxChromaDict() {
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

function createGeometry(edge1, edge2, reverseWinding = false) {
  if (edge1.length != edge2.length) {
    throw new Error("lengths r diff");
  }
 
  const positions = [];
  const indices = [];
  
  // add all non-null vertices to positions array, keeping edges in sync
  const vertexIndices1 = []; // map from original edge1 indices to position indices
  const vertexIndices2 = []; // map from original edge2 indices to position indices
  
  for (let i = 0; i < edge1.length; i++) {
    if (edge1[i] && edge2[i]) {
      vertexIndices1[i] = positions.length / 3;
      positions.push(edge1[i].x, edge1[i].y, edge1[i].z || 0);
      
      vertexIndices2[i] = positions.length / 3;
      positions.push(edge2[i].x, edge2[i].y, edge2[i].z || 0);
    } else {
      // if at least one vertex is null - skip both
      vertexIndices1[i] = null;
      vertexIndices2[i] = null;
    }
  }
  
  for (let i = 0; i < edge1.length - 1; i++) {
    const v00 = vertexIndices1[i];
    const v01 = vertexIndices1[i + 1];
    const v10 = vertexIndices2[i];
    const v11 = vertexIndices2[i + 1];
    
    // skip if any vertex is null (mesh will be discontinuous)
    if (v00 === null || v01 === null || v10 === null || v11 === null) {
      continue;
    }
    
    // create two triangles for this quad
    if (reverseWinding) {
      indices.push(v00, v10, v01);
      indices.push(v01, v10, v11);
    } else {
      indices.push(v00, v01, v10);
      indices.push(v01, v11, v10);
    }
  }
  
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
  
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

export { Slicer };
