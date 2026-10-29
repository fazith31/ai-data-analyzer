// Progressive Web App (PWA) Controller & Universal Mobile Installer
(function() {
  'use strict';

  // 1. Register Service Worker
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/sw.js')
        .then(function(reg) {
          console.log('PWA ServiceWorker registered with scope:', reg.scope);
        })
        .catch(function(err) {
          navigator.serviceWorker.register('/static/sw.js')
            .catch(function(e) { console.warn('PWA ServiceWorker fallback failed:', e); });
        });
    });
  }

  // 2. Check Standalone mode
  var isStandalone = window.matchMedia('(display-mode: standalone)').matches || 
                     window.navigator.standalone === true ||
                     document.referrer.includes('android-app://');

  if (isStandalone) {
    document.documentElement.classList.add('pwa-standalone');
    console.log('App running in native standalone window mode');
    return; // Don't show any install buttons or banners if already installed!
  }

  var deferredPrompt = null;
  var isIos = /iphone|ipad|ipod/.test(window.navigator.userAgent.toLowerCase());
  var isAndroid = /android/.test(window.navigator.userAgent.toLowerCase());

  // Listen for browser install prompt
  window.addEventListener('beforeinstallprompt', function(e) {
    e.preventDefault();
    deferredPrompt = e;
  });

  window.addEventListener('appinstalled', function() {
    deferredPrompt = null;
    document.documentElement.classList.add('pwa-standalone');
    var banner = document.getElementById('pwaBottomBanner');
    if (banner) banner.remove();
    var pill = document.getElementById('pwaNavInstallPill');
    if (pill) pill.remove();
    document.querySelectorAll('.pwa-install-btn').forEach(function(b) { b.style.display = 'none'; });
  });

  // 3. Setup Install UI on DOM ready
  document.addEventListener('DOMContentLoaded', initInstallUI);
  if (document.readyState === 'interactive' || document.readyState === 'complete') {
    initInstallUI();
  }

  var initialized = false;
  function initInstallUI() {
    if (initialized) return;
    initialized = true;

    // Unhide all existing template install buttons
    document.querySelectorAll('.pwa-install-btn').forEach(function(btn) {
      btn.style.display = 'inline-flex';
    });

    // A. Add Install Pill to Mobile Navbar (Visible immediately on phone screen!)
    var navbar = document.querySelector('.navbar');
    var navToggle = document.querySelector('.nav-toggle');
    if (navbar && !document.getElementById('pwaNavInstallPill')) {
      var pill = document.createElement('button');
      pill.type = 'button';
      pill.id = 'pwaNavInstallPill';
      pill.className = 'nav-install-pill';
      pill.innerHTML = '📲 Install App';
      pill.title = 'Install AI Data Analyzer on this device';
      pill.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        triggerInstall();
      });

      if (navToggle) {
        navbar.insertBefore(pill, navToggle);
      } else {
        navbar.appendChild(pill);
      }
    }

    // B. Inject Floating Bottom Install Banner
    if (!document.getElementById('pwaBottomBanner')) {
      var banner = document.createElement('div');
      banner.id = 'pwaBottomBanner';
      banner.className = 'pwa-bottom-banner';
      banner.innerHTML = [
        '<img src="/static/icons/icon-192.png" alt="App Icon" class="pwa-banner-icon">',
        '<div class="pwa-banner-text">',
        '  <strong>Install AI Data Analyzer</strong>',
        '  <span>Add to home screen for real app experience</span>',
        '</div>',
        '<button type="button" class="pwa-banner-install-btn" id="pwaBannerInstallBtn">Install</button>',
        '<button type="button" class="pwa-banner-close-btn" id="pwaBannerCloseBtn" aria-label="Close">✕</button>'
      ].join('');

      document.body.appendChild(banner);

      // Slide banner in after 800ms
      setTimeout(function() {
        if (!document.documentElement.classList.contains('pwa-standalone')) {
          banner.classList.add('visible');
        }
      }, 800);

      document.getElementById('pwaBannerInstallBtn').addEventListener('click', function(e) {
        e.preventDefault();
        triggerInstall();
      });

      document.getElementById('pwaBannerCloseBtn').addEventListener('click', function() {
        banner.classList.remove('visible');
        setTimeout(function() { banner.remove(); }, 400);
      });
    }

    // C. Inject Install Instructions Modal
    if (!document.getElementById('pwaModalOverlay')) {
      var modal = document.createElement('div');
      modal.id = 'pwaModalOverlay';
      modal.className = 'pwa-modal-overlay';
      modal.innerHTML = [
        '<div class="pwa-modal-card">',
        '  <h3 id="pwaModalTitle">📲 Install AI Data Analyzer</h3>',
        '  <div id="pwaModalContent"></div>',
        '  <button type="button" class="pwa-modal-close-btn" id="pwaModalCloseBtn">Got it</button>',
        '</div>'
      ].join('');

      document.body.appendChild(modal);

      document.getElementById('pwaModalCloseBtn').addEventListener('click', function() {
        modal.classList.remove('open');
      });
      modal.addEventListener('click', function(e) {
        if (e.target === modal) modal.classList.remove('open');
      });
    }

    // Bind all buttons with class .pwa-install-btn
    document.querySelectorAll('.pwa-install-btn').forEach(function(btn) {
      btn.onclick = function(e) {
        e.preventDefault();
        triggerInstall();
      };
    });
  }

  // 4. Trigger Installation / Show Clear Instructions
  function triggerInstall() {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then(function(choice) {
        if (choice.outcome === 'accepted') {
          console.log('User accepted install');
        }
        deferredPrompt = null;
      });
      return;
    }

    // Show Guide Modal
    var modal = document.getElementById('pwaModalOverlay');
    var content = document.getElementById('pwaModalContent');
    if (!modal || !content) return;

    if (isIos) {
      content.innerHTML = [
        '<p style="margin:0 0 10px 0; color:#94a3b8; font-size:13px;">To install on iPhone or iPad:</p>',
        '<ol>',
        '  <li>Tap the <strong>Share</strong> button <span style="font-size:16px;">⎋</span> at the bottom of Safari</li>',
        '  <li>Scroll down and tap <strong>"Add to Home Screen"</strong> ➕</li>',
        '  <li>Tap <strong>Add</strong> in the top right</li>',
        '</ol>',
        '<p style="margin:0; font-size:12px; color:#10b981;">The app will install directly onto your home screen!</p>'
      ].join('');
    } else {
      // Android / Chrome / Edge
      content.innerHTML = [
        '<p style="margin:0 0 10px 0; color:#94a3b8; font-size:13px;">To install on Android or Desktop:</p>',
        '<ol>',
        '  <li>Tap the browser menu <strong>⋮ (three dots)</strong> in the top-right</li>',
        '  <li>Select <strong>"Install app"</strong> or <strong>"Add to Home screen"</strong></li>',
        '  <li>Confirm <strong>"Install"</strong></li>',
        '</ol>',
        '<p style="margin:0; font-size:12px; color:#10b981;">✓ Launches in full screen like a native mobile app!</p>'
      ].join('');
    }

    modal.classList.add('open');
  }

})();
