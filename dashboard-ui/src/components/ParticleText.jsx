import React, { useEffect, useRef } from 'react';

export default function ParticleText({
  text = 'Future Interfaces',
  particleSize = 2.0,
  density = 3,
  color = '#f8fafc',
  highlightColor = '#a855f7',
  scatter = 160,
  gatherDuration = 1400,
  stagger = 300,
  pointerRepel = 35,
  repelRadius = 100,
  idleDrift = 0.5,
  fontWeight = 800,
  fontFamily = 'Inter, -apple-system, sans-serif',
  glow = true,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId;
    let particles = [];
    let mouse = { x: -9999, y: -9999 };

    const updateSize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      initParticles(rect.width, rect.height);
    };

    const initParticles = (w, h) => {
      if (!w || !h) return;

      const offscreen = document.createElement('canvas');
      offscreen.width = w;
      offscreen.height = h;
      const offCtx = offscreen.getContext('2d');
      if (!offCtx) return;

      // Calculate font size to fit container width cleanly
      const targetFontSize = Math.max(36, Math.min(w / (text.length * 0.62), h * 0.55));
      offCtx.fillStyle = '#ffffff';
      offCtx.font = `${fontWeight} ${targetFontSize}px ${fontFamily}`;
      offCtx.textAlign = 'center';
      offCtx.textBaseline = 'middle';
      offCtx.fillText(text, w / 2, h / 2);

      const imageData = offCtx.getImageData(0, 0, w, h);
      const data = imageData.data;
      particles = [];

      const step = Math.max(2, Math.floor(density));

      for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
          const index = (y * w + x) * 4;
          const alpha = data[index + 3];

          if (alpha > 120) {
            const targetX = x;
            const targetY = y;

            const angle = Math.random() * Math.PI * 2;
            const dist = (Math.random() * 0.6 + 0.4) * scatter;
            const initialX = targetX + Math.cos(angle) * dist;
            const initialY = targetY + Math.sin(angle) * dist;

            const isHighlight = Math.random() < 0.35;

            particles.push({
              x: initialX,
              y: initialY,
              initialX: initialX,
              initialY: initialY,
              targetX: targetX,
              targetY: targetY,
              size: particleSize * (isHighlight ? 1.2 : 0.95),
              color: isHighlight ? highlightColor : color,
              glow: isHighlight && glow,
              delay: Math.random() * stagger,
            });
          }
        }
      }
    };

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };

    const handleMouseLeave = () => {
      mouse.x = -9999;
      mouse.y = -9999;
    };

    window.addEventListener('resize', updateSize);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    updateSize();

    const startTime = performance.now();

    const render = (now) => {
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      ctx.clearRect(0, 0, w, h);

      const elapsed = now - startTime;

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const pElapsed = Math.max(0, elapsed - p.delay);
        const progress = Math.min(1, pElapsed / gatherDuration);

        // Smooth cubic ease out
        const ease = 1 - Math.pow(1 - progress, 3);
        const currentTargetX = p.initialX + (p.targetX - p.initialX) * ease;
        const currentTargetY = p.initialY + (p.targetY - p.initialY) * ease;

        // Pointer Repulsion Effect
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        let repelX = 0;
        let repelY = 0;
        if (dist < repelRadius && dist > 0) {
          const force = (1 - dist / repelRadius) * pointerRepel;
          repelX = -(dx / dist) * force;
          repelY = -(dy / dist) * force;
        }

        // Idle drift
        const driftX = Math.sin(now * 0.0012 + i * 0.1) * idleDrift;
        const driftY = Math.cos(now * 0.0012 + i * 0.1) * idleDrift;

        const destX = currentTargetX + repelX + driftX;
        const destY = currentTargetY + repelY + driftY;

        p.x += (destX - p.x) * 0.14;
        p.y += (destY - p.y) * 0.14;

        ctx.fillStyle = p.color;
        if (p.glow) {
          ctx.shadowColor = p.color;
          ctx.shadowBlur = 6;
        } else {
          ctx.shadowBlur = 0;
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', updateSize);
      canvas.removeEventListener('mousemove', handleMouseMove);
      canvas.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, [text, particleSize, density, color, highlightColor, scatter, gatherDuration, stagger, pointerRepel, repelRadius, idleDrift, glow]);

  return (
    <div style={{ width: '100%', height: '100%', minHeight: 180, position: 'relative', overflow: 'hidden' }}>
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
    </div>
  );
}
