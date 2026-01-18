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
    // mouse events
    this.handle1.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle1")
    );
    this.handle2.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle2")
    );
    document.addEventListener("mousemove", (e) => this.drag(e));
    document.addEventListener("mouseup", () => this.endDrag());

    // touch events
    this.handle1.addEventListener("touchstart", (e) =>
      this.startDrag(e, "handle1")
    );
    this.handle2.addEventListener("touchstart", (e) =>
      this.startDrag(e, "handle2")
    );
    document.addEventListener("touchmove", (e) => this.drag(e));
    document.addEventListener("touchend", () => this.endDrag());
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

    // get clientX and clientY from either mouse or touch event
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const rect = this.container.getBoundingClientRect();
    const x = clientX - rect.left - this.centerX;
    const y = clientY - rect.top - this.centerY;

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

  setHueRange(start, end) {
    this.angle1 = start;
    this.angle2 = end;
    this.updateDisplay();
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
    this.col2 = "#ff0000";

    this.isDragging = false;
    this.activeHandle = null;
    this.onChange = null;

    this.lastHandle = null; // handle which was last touched

    this.track.style.background = gradientCSS;
    this.init();
    this.updateDisplay();
  }

  init() {
    // mouse events
    this.handle1.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle1")
    );
    this.handle2.addEventListener("mousedown", (e) =>
      this.startDrag(e, "handle2")
    );
    document.addEventListener("mousemove", (e) => this.drag(e));
    document.addEventListener("mouseup", () => this.endDrag());

    // touch events
    this.handle1.addEventListener("touchstart", (e) =>
      this.startDrag(e, "handle1")
    );
    this.handle2.addEventListener("touchstart", (e) =>
      this.startDrag(e, "handle2")
    );
    document.addEventListener("touchmove", (e) => this.drag(e));
    document.addEventListener("touchend", () => this.endDrag());
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

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;

    let percent = (clientX - rect.left) / rect.width;
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

  setValues(start, end) {
    this.value1 = start;
    this.value2 = end;
    this.updateDisplay();
  }

  setGradient(col1, col2) {
    this.track.style.background = `linear-gradient(to right, ${col1}, ${col2})`;
  }
}

export { CircularSlider, TwoHandleSlider };
