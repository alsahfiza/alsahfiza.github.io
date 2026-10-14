(() => {
  let loading;
  function renderMath() {
    const main = document.getElementById('_main');
    if (!main || main.dataset.mathjax !== 'true') return;
    if (!loading) {
      loading = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.id = '_MathJax';
        script.src = window._baseURL + 'assets/bower_components/MathJax/es5/tex-mml-chtml.js';
        script.onload = resolve;
        script.onerror = () => {
          script.remove();
          reject(new Error('MathJax failed to load'));
        };
        document.head.appendChild(script);
      });
    }
    loading.then(() => {
      if (window.MathJax && MathJax.typesetPromise) {
        const ready = MathJax.startup && MathJax.startup.promise || Promise.resolve();
        return ready.then(() => {
          const current = document.getElementById('_main');
          if (current && current.dataset.mathjax === 'true') {
            current.querySelectorAll('script[type^="math/tex"]').forEach(equation => {
              const display = /mode\s*=\s*display/.test(equation.type);
              const delimiters = display ? ['$$', '$$'] : ['\\(', '\\)'];
              equation.replaceWith(document.createTextNode(
                delimiters[0] + equation.textContent + delimiters[1]
              ));
            });
            return MathJax.typesetPromise([current]);
          }
        });
      }
    }).catch(() => { loading = undefined; });
  }
  renderMath();
  const pushState = document.getElementById('_pushState');
  if (pushState) pushState.addEventListener('hy-push-state-load', renderMath);
})();
