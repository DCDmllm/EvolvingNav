(() => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const ambient = [...document.querySelectorAll('.ambient-video')];
  const toggle = document.getElementById('motion-toggle');
  let motionPaused = reducedMotion.matches;
  let heroVisible = true;
  const syncMotion = () => {
    document.body.classList.toggle('motion-paused', motionPaused);
    toggle.setAttribute('aria-pressed', String(motionPaused));
    toggle.innerHTML = motionPaused ? 'Play background <span aria-hidden="true">▶</span>' : 'Pause background <span aria-hidden="true">Ⅱ</span>';
    ambient.forEach(video => {
      if (motionPaused || !heroVisible || document.hidden) video.pause();
      else video.play().catch(() => {});
    });
  };
  toggle.addEventListener('click', () => { motionPaused = !motionPaused; syncMotion(); });
  reducedMotion.addEventListener('change', event => { motionPaused = event.matches; syncMotion(); });
  document.addEventListener('visibilitychange', syncMotion);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => { heroVisible = entries[0].isIntersecting; syncMotion(); }).observe(document.querySelector('.cinema-hero'));
    const demos = new IntersectionObserver(entries => entries.forEach(({target,isIntersecting}) => {
      if (!isIntersecting) target.pause();
      else if (!reducedMotion.matches) target.play().catch(() => {});
    }), {threshold:0.2});
    document.querySelectorAll('.hssd-video-frame video').forEach(video => {
      video.removeAttribute('autoplay');
      video.setAttribute('aria-label', video.closest('figure').querySelector('.hssd-video-label').textContent);
      demos.observe(video);
    });
  }
  syncMotion();
  document.getElementById('copy-citation').addEventListener('click', async () => {
    const status = document.getElementById('copy-status');
    try { await navigator.clipboard.writeText(document.querySelector('#bibtex pre').textContent); status.textContent = 'Copied'; }
    catch { status.textContent = 'Select the citation below to copy.'; }
  });
})();
