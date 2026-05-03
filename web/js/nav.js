// =========================================
// NAVIGATION UNDERLINE
// Shared across all pages (index, report, about)
// =========================================
document.addEventListener('DOMContentLoaded', () => {
  const underline = document.querySelector('.underline');
  const navLinks  = document.querySelectorAll('.nav-links li a');

  if (!underline || !navLinks.length) return;

  function moveUnderline(link) {
    const rect    = link.getBoundingClientRect();
    const navRect = link.closest('.top-nav').getBoundingClientRect();
    underline.style.width = rect.width + "px";
    underline.style.left  = (rect.left - navRect.left) + "px";
  }

  let activeFound = false;

  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    const path = window.location.pathname;

    // Match by filename at end of path
    if (href && (path.endsWith(href) || path.endsWith('/' + href))) {
      link.classList.add('active');
      moveUnderline(link);
      activeFound = true;
    }

    link.addEventListener('click', () => {
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      moveUnderline(link);
    });
  });

  // Fallback: if no match found (e.g. root path = index.html)
  if (!activeFound && navLinks.length > 0) {
    navLinks[0].classList.add('active');
    moveUnderline(navLinks[0]);
  }
});
