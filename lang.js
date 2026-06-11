// Simple ES/EN toggle for SMM pages.
// Any element with data-es="..." and data-en="..." attributes will have its
// innerHTML swapped when the user clicks the #lang-toggle button.
// Pages can define window.applyChartLang(lang) to update embedded charts.
(function () {
  function applyLang(lang) {
    document.documentElement.lang = lang;

    document.querySelectorAll('[data-es]').forEach(function (el) {
      var val = el.getAttribute('data-' + lang);
      if (val !== null) {
        el.innerHTML = val;
      }
    });

    var btn = document.getElementById('lang-toggle');
    if (btn) {
      btn.textContent = (lang === 'es') ? 'English' : 'Español';
    }

    if (typeof window.applyChartLang === 'function') {
      window.applyChartLang(lang);
    }

    try {
      localStorage.setItem('smm_lang', lang);
    } catch (e) {
      /* ignore (e.g. privacy mode) */
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    var saved = 'es';
    try {
      saved = localStorage.getItem('smm_lang') || 'es';
    } catch (e) {
      saved = 'es';
    }

    applyLang(saved);

    var btn = document.getElementById('lang-toggle');
    if (btn) {
      btn.addEventListener('click', function () {
        var next = (document.documentElement.lang === 'es') ? 'en' : 'es';
        applyLang(next);
      });
    }
  });
})();
