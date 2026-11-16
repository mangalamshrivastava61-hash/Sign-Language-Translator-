// main.js - Sign Language Translator frontend logic
(function () {
  "use strict";

  var cameraImg        = document.getElementById("cameraFeed");
  var cameraPlaceholder = document.getElementById("cameraPlaceholder");
  var cameraFrame      = document.getElementById("cameraFrame");
  var signDisplay      = document.getElementById("detectedSign");
  var confidenceFill   = document.getElementById("confidenceFill");
  var confidenceValue  = document.getElementById("confidenceValue");
  var statusDot        = document.getElementById("statusDot");
  var statusText       = document.getElementById("statusText");
  var sentenceText     = document.getElementById("sentenceText");
  var wordCount        = document.getElementById("wordCount");
  var btnClear         = document.getElementById("btnClear");
  var btnDeleteLast    = document.getElementById("btnDeleteLast");
  var btnCopy          = document.getElementById("btnCopy");
  var btnHelp          = document.getElementById("btnHelp");
  var modalOverlay     = document.getElementById("modalOverlay");
  var modalClose       = document.getElementById("modalClose");

  // Start MJPEG camera feed
  function startFeed() {
    cameraImg.src = "/video_feed?" + Date.now();
    cameraImg.onload = function () {
      cameraPlaceholder.classList.add("hidden");
    };
    cameraImg.onerror = function () {
      cameraPlaceholder.textContent = "Camera unavailable";
    };
  }

  // Poll /status endpoint
  var pollTimer = null;

  function poll() {
    fetch("/status")
      .then(function (resp) {
        if (!resp.ok) { schedulePoll(2000); return; }
        return resp.json();
      })
      .then(function (d) {
        if (!d) return;

        // Detected sign
        if (d.sign) {
          signDisplay.textContent = d.sign;
          signDisplay.classList.remove("empty");
          cameraFrame.classList.add("detecting");
        } else {
          signDisplay.textContent = "Awaiting hand gesture...";
          signDisplay.classList.add("empty");
          cameraFrame.classList.remove("detecting");
        }

        // Confidence
        var pct = Math.round((d.confidence || 0) * 100);
        confidenceFill.style.width = pct + "%";
        confidenceValue.textContent = pct + "%";

        // Status
        var st = (d.state || "Ready").toLowerCase();
        statusText.textContent = d.state || "Ready";
        statusDot.className = "status-dot";
        if (st.indexOf("analyz") !== -1)    statusDot.classList.add("analyzing");
        else if (st.indexOf("cool") !== -1 || st.indexOf("pause") !== -1) statusDot.classList.add("cooldown");
        else if (st.indexOf("error") !== -1) statusDot.classList.add("error");

        // Sentence
        if (d.sentence) {
          sentenceText.textContent = d.sentence;
          sentenceText.classList.remove("empty");
        } else {
          sentenceText.textContent = "Recognized words will construct here in real-time...";
          sentenceText.classList.add("empty");
        }
        wordCount.textContent = (d.word_count || 0) + " words";

        schedulePoll(400);
      })
      .catch(function () {
        schedulePoll(2000);
      });
  }

  function schedulePoll(ms) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(poll, ms);
  }

  // Clear button
  if (btnClear) {
    btnClear.addEventListener("click", function () {
      fetch("/clear", { method: "POST" });
    });
  }

  // Undo last button
  if (btnDeleteLast) {
    btnDeleteLast.addEventListener("click", function () {
      fetch("/delete_last", { method: "POST" });
    });
  }

  // Copy to clipboard
  if (btnCopy) {
    btnCopy.addEventListener("click", function () {
      var text = sentenceText.textContent;
      if (!text || sentenceText.classList.contains("empty")) return;
      navigator.clipboard.writeText(text).then(function () {
        btnCopy.classList.add("copied");
        setTimeout(function () {
          btnCopy.classList.remove("copied");
        }, 1500);
      });
    });
  }

  // Keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.code === "Space") {
      e.preventDefault();
      fetch("/add_sign", { method: "POST" });
    } else if (e.code === "Backspace") {
      e.preventDefault();
      fetch("/delete_last", { method: "POST" });
    }
  });

  // Help modal
  if (btnHelp && modalOverlay && modalClose) {
    btnHelp.addEventListener("click", function () {
      modalOverlay.classList.add("open");
    });
    modalClose.addEventListener("click", function () {
      modalOverlay.classList.remove("open");
    });
    modalOverlay.addEventListener("click", function (e) {
      if (e.target === modalOverlay) modalOverlay.classList.remove("open");
    });
  }

  // Init
  startFeed();
  poll();
})();
