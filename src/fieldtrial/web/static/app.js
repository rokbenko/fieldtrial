// fieldtrial console: timer, undo countdown, live refresh (SSE) and keyboard / foot pedal.
// No inline scripts and no eval, so the page runs under a strict Content-Security-Policy.
"use strict";

(function () {
  const DEFAULT_KEYS = { start_stop: " ", confirm: "Enter", invalid: "i", undo: "u" };
  const STORAGE_KEY = "fieldtrial.keys";
  let clockOffset = 0; // server time minus this device's time, in ms
  let rebinding = null; // the key button waiting for a new key
  let dirty = false; // the operator has started filling in a form

  // --- keys ------------------------------------------------------------------------------

  function loadKeys() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      return Object.assign({}, DEFAULT_KEYS, saved);
    } catch (e) {
      return Object.assign({}, DEFAULT_KEYS);
    }
  }

  function saveKeys(keys) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
    } catch (e) {
      /* private mode: keys last for this page only */
    }
  }

  function keyName(key) {
    return key === " " ? "Space" : key;
  }

  function showKeys() {
    const keys = loadKeys();
    document.querySelectorAll("button.key[data-bind]").forEach(function (button) {
      button.textContent = keyName(keys[button.dataset.bind] || "");
      button.classList.remove("listening");
    });
  }

  function typing(target) {
    if (!target || !target.tagName) return false;
    const tag = target.tagName.toLowerCase();
    if (tag === "textarea" || tag === "select" || target.isContentEditable) return true;
    if (tag !== "input") return false;
    return !["radio", "checkbox", "button", "submit"].includes(target.type);
  }

  function click(selector) {
    const element = document.querySelector("#panel " + selector);
    if (!element) return false;
    if (element.tagName.toLowerCase() === "summary") {
      const details = element.parentElement;
      details.open = true;
      const input = details.querySelector("input[name=reason]");
      if (input) input.focus();
      return true;
    }
    const form = element.closest("form");
    if (form && element.type === "submit") {
      // Make sure HTMX has wired up a form that was just swapped in.
      if (window.htmx) window.htmx.process(form);
      form.requestSubmit(element);
    } else {
      element.click();
    }
    return true;
  }

  function chooseStage(digit) {
    const radio = document.querySelector('#panel input[name=stage][data-key="' + digit + '"]');
    if (!radio) return false;
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  document.addEventListener("keydown", function (event) {
    if (rebinding) {
      event.preventDefault();
      const keys = loadKeys();
      if (event.key !== "Escape") keys[rebinding.dataset.bind] = event.key;
      saveKeys(keys);
      rebinding = null;
      showKeys();
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey || typing(event.target)) return;
    const keys = loadKeys();
    let handled = false;
    if (/^[0-9]$/.test(event.key)) handled = chooseStage(event.key);
    else if (event.key === keys.start_stop) handled = click('[data-action="start_stop"]');
    else if (event.key === keys.confirm) handled = click('[data-action="confirm"]');
    else if (event.key.toLowerCase() === String(keys.invalid).toLowerCase())
      handled = click('[data-action="invalid"]');
    else if (event.key.toLowerCase() === String(keys.undo).toLowerCase())
      handled = click('[data-action="undo"]');
    if (handled) event.preventDefault();
  });

  document.addEventListener("click", function (event) {
    const button = event.target.closest("button.key[data-bind]");
    if (button) {
      showKeys();
      rebinding = button;
      button.textContent = "press a key…";
      button.classList.add("listening");
      return;
    }
    if (event.target.closest("[data-reset-keys]")) {
      saveKeys({});
      showKeys();
    }
  });

  // Choosing the success stage pre-selects "success" as the reason it ended.
  document.addEventListener("change", function (event) {
    const target = event.target;
    if (target.closest && target.closest("#panel form")) dirty = true;
    if (target.name !== "stage") return;
    const form = target.closest("form.label-form");
    if (!form) return;
    const success = form.dataset.successIndex;
    const chosen = form.querySelector("input[name=termination]:checked");
    const successRadio = form.querySelector('input[name=termination][value="success"]');
    if (target.value === success && !chosen && successRadio) successRadio.checked = true;
    if (target.value !== success && chosen === successRadio && successRadio) successRadio.checked = false;
  });
  document.addEventListener("input", function (event) {
    if (event.target.closest && event.target.closest("#panel form")) dirty = true;
  });

  // --- timer and undo --------------------------------------------------------------------

  function now() {
    return Date.now() + clockOffset;
  }

  function format(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return minutes + ":" + String(seconds).padStart(2, "0");
  }

  function tick() {
    document.querySelectorAll(".timer[data-started-at]").forEach(function (timer) {
      const elapsed = now() - Date.parse(timer.dataset.startedAt);
      timer.textContent = format(elapsed);
      const timeout = parseFloat(timer.dataset.timeout) * 1000;
      if (timeout > 0) {
        timer.classList.toggle("warn", elapsed >= 0.8 * timeout && elapsed < timeout);
        timer.classList.toggle("over", elapsed >= timeout);
      }
    });
    document.querySelectorAll("form.undo[data-undo-until]").forEach(function (form) {
      const left = Date.parse(form.dataset.undoUntil) - now();
      if (left <= 0) {
        form.remove();
        return;
      }
      const countdown = form.querySelector(".countdown");
      if (countdown) countdown.textContent = "(" + Math.ceil(left / 1000) + " s)";
    });
  }

  function syncClock() {
    const panel = document.querySelector(".panel[data-server-now]");
    if (panel) clockOffset = Date.parse(panel.dataset.serverNow) - Date.now();
  }

  // --- live refresh ----------------------------------------------------------------------

  let pending = null;

  function refresh() {
    if (dirty) return; // never wipe a form the operator is filling in
    if (pending) clearTimeout(pending);
    pending = setTimeout(function () {
      pending = null;
      if (window.htmx) window.htmx.trigger(document.body, "ft-refresh");
    }, 150);
  }

  function listen() {
    const panel = document.getElementById("panel");
    if (!panel || !panel.dataset.events || !window.EventSource) return;
    const source = new EventSource(panel.dataset.events);
    source.addEventListener("changed", function (event) {
      // Skip events the panel already shows (usually this device's own action).
      const shown = document.querySelector(".panel[data-last-event]");
      if (shown && event.lastEventId && event.lastEventId <= shown.dataset.lastEvent) return;
      refresh();
    });
  }

  document.addEventListener("htmx:afterSwap", function (event) {
    if (event.target && event.target.id === "panel") {
      dirty = false;
      syncClock();
      tick();
      const flash = document.getElementById("flash");
      if (flash && !event.detail.failed) flash.textContent = "";
    }
  });

  document.addEventListener("DOMContentLoaded", function () {
    syncClock();
    showKeys();
    tick();
    setInterval(tick, 250);
    listen();
  });
})();
