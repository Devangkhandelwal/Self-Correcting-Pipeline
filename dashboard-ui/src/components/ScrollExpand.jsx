import React, { useState, useEffect, useRef } from 'react';

export default function ScrollExpand({
  src = '/hero.jpg',
  alt = 'Product hero',
  title = 'Self-Correcting Data Pipeline',
  scrollHint = 'Scroll down inside the frame to expand',
  useWindowScroll = true,
  startWidth = 50,
  startHeight = 52,
  startRadius = 24,
  endRadius = 0,
  mediaZoom = 1.3,
  overlayScrim = 0.45,
  enabled = true,
  children,
}) {
  const containerRef = useRef(null);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    if (!enabled) return;

    const handleScroll = () => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const windowHeight = window.innerHeight;

      // Start calculating progress when the container top moves into view near navbar
      // At initial page load (top of page), rect.top is ~120px, so progress should be 0.
      const initialTop = 120; 
      const scrollableRange = windowHeight * 0.7;
      const scrolled = (windowHeight - rect.top) - (windowHeight - initialTop);

      let progress = scrolled / scrollableRange;
      progress = Math.max(0, Math.min(1, progress));

      setScrollProgress(progress);
    };

    if (useWindowScroll) {
      window.addEventListener('scroll', handleScroll, { passive: true });
      handleScroll();
      return () => window.removeEventListener('scroll', handleScroll);
    }
  }, [useWindowScroll, enabled]);

  // Compute interpolated visual states
  const currentWidth = startWidth + (100 - startWidth) * scrollProgress;
  const currentHeight = startHeight + (100 - startHeight) * scrollProgress;
  const currentRadius = startRadius - (startRadius - endRadius) * scrollProgress;
  const currentZoom = mediaZoom - (mediaZoom - 1.0) * scrollProgress;
  
  // Title stays 100% visible at top, then fades out as user scrolls
  const titleOpacity = Math.max(0, 1 - scrollProgress * 1.8);
  const childrenOpacity = Math.max(0, (scrollProgress - 0.25) / 0.75);

  return (
    <div
      ref={containerRef}
      className="scroll-expand-wrapper"
      style={{
        position: 'relative',
        width: '100%',
        minHeight: '130vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        paddingTop: '1rem',
        paddingBottom: '6rem',
      }}
    >
      {/* Sticky/Expanding media container */}
      <div
        className="scroll-expand-frame"
        style={{
          position: 'sticky',
          top: '5rem',
          width: `${currentWidth}%`,
          height: `${currentHeight}vh`,
          borderRadius: `${currentRadius}px`,
          overflow: 'hidden',
          transition: 'width 0.1s linear, height 0.1s linear, border-radius 0.1s linear',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.7), 0 0 50px rgba(6, 182, 212, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.15)',
          zIndex: 10,
        }}
      >
        {/* Hero Image with Zoom */}
        <img
          src={src}
          alt={alt}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: `scale(${currentZoom})`,
            transition: 'transform 0.1s linear',
          }}
        />

        {/* Dark Scrim Overlay */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            backgroundColor: `rgba(3, 7, 18, ${overlayScrim})`,
            pointerEvents: 'none',
          }}
        />

        {/* Hero Title & Scroll Hint Overlay */}
        {title && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: titleOpacity,
              transition: 'opacity 0.15s ease-out',
              pointerEvents: 'none',
              padding: '2rem',
              textAlign: 'center',
              zIndex: 12,
            }}
          >
            <h1
              style={{
                fontSize: 'clamp(2.2rem, 5.5vw, 4.2rem)',
                fontWeight: 900,
                color: '#FFFFFF',
                letterSpacing: '-0.02em',
                textShadow: '0 4px 30px rgba(0, 0, 0, 0.9), 0 0 20px rgba(99, 102, 241, 0.5)',
                margin: 0,
                lineHeight: 1.1,
              }}
            >
              {title}
            </h1>

            {scrollHint && (
              <div
                style={{
                  marginTop: '1.5rem',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  color: '#F8FAFC',
                  background: 'rgba(15, 23, 42, 0.85)',
                  padding: '0.6rem 1.4rem',
                  borderRadius: '9999px',
                  border: '1px solid rgba(99, 102, 241, 0.4)',
                  boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5), 0 0 16px rgba(99, 102, 241, 0.3)',
                  backdropFilter: 'blur(12px)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                <span style={{ fontSize: '1.1rem' }}>👇</span>
                <span>{scrollHint}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Children content that fades in as frame expands */}
      {children && (
        <div
          className="scroll-expand-content"
          style={{
            width: '100%',
            maxWidth: '1280px',
            marginTop: '3rem',
            opacity: childrenOpacity,
            transform: `translateY(${(1 - childrenOpacity) * 30}px)`,
            transition: 'opacity 0.2s ease-out, transform 0.2s ease-out',
            zIndex: 15,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}
