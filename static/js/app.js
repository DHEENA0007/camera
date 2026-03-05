/**
 * CCTV Vehicle Monitor - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {

    // === DateTime Display ===
    function updateDateTime() {
        const el = document.getElementById('datetime-display');
        if (!el) return;
        const now = new Date();
        const options = {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        };
        el.textContent = now.toLocaleString('en-IN', options);
    }
    updateDateTime();
    setInterval(updateDateTime, 1000);

    // === Sidebar Toggle (Mobile) ===
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.getElementById('sidebar');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });

        // Close sidebar on outside click (mobile)
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                    sidebar.classList.remove('open');
                }
            }
        });
    }

    // === Chart Bar Scaling ===
    const chartBars = document.querySelectorAll('.chart-bar');
    if (chartBars.length > 0) {
        let maxVal = 0;
        chartBars.forEach(bar => {
            const val = parseInt(bar.getAttribute('data-value') || '0');
            if (val > maxVal) maxVal = val;
        });

        if (maxVal > 0) {
            chartBars.forEach(bar => {
                const val = parseInt(bar.getAttribute('data-value') || '0');
                const pct = Math.max(5, (val / maxVal) * 100);
                bar.style.height = pct + '%';
            });
        }
    }

    // === System Status ===
    const statusDot = document.getElementById('system-status-dot');
    const statusText = document.getElementById('system-status-text');

    function checkSystemStatus() {
        fetch('/api/stats/')
            .then(r => {
                if (r.ok) {
                    if (statusDot) statusDot.style.background = 'var(--accent-success)';
                    if (statusText) statusText.textContent = 'System Active';
                }
            })
            .catch(() => {
                if (statusDot) statusDot.style.background = 'var(--accent-warning)';
                if (statusText) statusText.textContent = 'Checking...';
            });
    }
    checkSystemStatus();
    setInterval(checkSystemStatus, 30000);

    // === Table Row Hover Effect ===
    document.querySelectorAll('.table-row').forEach(row => {
        row.addEventListener('mouseenter', function() {
            this.style.background = 'rgba(99, 102, 241, 0.06)';
        });
        row.addEventListener('mouseleave', function() {
            this.style.background = '';
        });
    });

    // === Smooth number animation ===
    function animateNumber(el, target, duration = 800) {
        const start = parseInt(el.textContent) || 0;
        if (start === target) return;

        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (target - start) * eased);
            el.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }
        requestAnimationFrame(update);
    }

    // Animate stat card values on load
    document.querySelectorAll('.stat-card__value').forEach(el => {
        const val = parseInt(el.textContent) || 0;
        el.textContent = '0';
        setTimeout(() => animateNumber(el, val), 300);
    });

    document.querySelectorAll('.summary-card__value').forEach(el => {
        const val = parseInt(el.textContent) || 0;
        el.textContent = '0';
        setTimeout(() => animateNumber(el, val), 300);
    });

    // === Keyboard shortcuts ===
    document.addEventListener('keydown', function(e) {
        // Ctrl+K: Focus search
        if (e.ctrlKey && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('search-input');
            if (searchInput) {
                searchInput.focus();
            } else {
                window.location.href = '/search/';
            }
        }
    });

    console.log('%c🎥 CCTV Vehicle Monitor', 'font-size: 20px; font-weight: bold; color: #6366f1;');
    console.log('%cVehicle Detection & License Plate Recognition System', 'color: #8b92a8;');
});
