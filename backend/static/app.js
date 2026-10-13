/* Travel-Genie frontend: drives the TravelAgent workflow.
 * Endpoints: GET /api/health, POST /workflows, POST /workflows/{id}/requirements,
 * POST /workflows/{id}/clarifications, GET /workflows/{id}. It never calls POST /api/plan. */
(function () {
  "use strict";

  var form = document.getElementById("plan-form");
  var results = document.getElementById("results");
  var submitBtn = document.getElementById("submit-btn");
  var pill = document.getElementById("mode-pill");
  var discoverLink = document.getElementById("discover-link");
  var depart = document.getElementById("departure_date");
  var ret = document.getElementById("return_date");
  var duration = document.getElementById("duration_days");
  var durationHint = document.getElementById("duration-hint");
  var lastPayload = null;
  var workflowId = null;
  var loadingTimer = null;

  // ---------- small DOM helper (text is always set via textContent, never innerHTML) ----------
  function h(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "class") node.className = attrs[k];
        else if (k === "text") node.textContent = attrs[k];
        else node.setAttribute(k, attrs[k]);
      });
    }
    function add(c) {
      if (c == null || c === false) return;
      if (Array.isArray(c)) return c.forEach(add);
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    for (var i = 2; i < arguments.length; i++) add(arguments[i]);
    return node;
  }

  function money(n, currency) {
    try {
      return new Intl.NumberFormat(currency === "INR" ? "en-IN" : undefined, {
        style: "currency", currency: currency || "INR", maximumFractionDigits: 0
      }).format(n);
    } catch (e) {
      return (currency || "") + " " + Math.round(n).toLocaleString();
    }
  }

  function prettyDate(iso) {
    var d = new Date(iso + "T00:00:00");
    return isNaN(d) ? iso : d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
  }

  function isoDate(d) {
    var m = String(d.getMonth() + 1).padStart(2, "0");
    var day = String(d.getDate()).padStart(2, "0");
    return d.getFullYear() + "-" + m + "-" + day;
  }

  function titleCase(value) {
    return String(value || "").replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  // ---------- form defaults and date logic ----------
  (function setDefaults() {
    var start = new Date();
    start.setDate(start.getDate() + 14);
    depart.value = isoDate(start);
    depart.min = isoDate(new Date());
    ret.min = depart.value;
  })();

  function syncDates() {
    ret.min = depart.value || isoDate(new Date());
    if (ret.value && depart.value) {
      var days = Math.round((new Date(ret.value) - new Date(depart.value)) / 86400000) + 1;
      if (days >= 1) duration.value = days;
      duration.disabled = true;
      durationHint.textContent = "Worked out from your dates.";
    } else {
      duration.disabled = false;
      durationHint.textContent = "Used when no return date is set.";
    }
  }
  depart.addEventListener("change", syncDates);
  ret.addEventListener("change", syncDates);

  // ---------- health pill ----------
  fetch("/api/health")
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var ai = d.integrations.gemini.configured;
      var groq = d.integrations.groq && d.integrations.groq.configured;
      var search = d.integrations.serpapi.configured;
      if (ai || groq || search) {
        pill.textContent = "Keys found: " + [ai ? "Gemini" : null, groq ? "Groq backup" : null, search ? "SerpApi" : null].filter(Boolean).join(" and ") + " (checked on first plan)";
        pill.classList.remove("demo");
      } else {
        pill.textContent = "Demo mode: no API keys set";
        pill.classList.add("demo");
      }
    })
    .catch(function () { pill.textContent = "Server not reachable"; pill.classList.add("demo"); });

  // ---------- payload ----------
  function readPayload() {
    var fd = new FormData(form);
    var payload = {
      origin: (fd.get("origin") || "").trim(),
      destination: (fd.get("destination") || "").trim(),
      departure_date: fd.get("departure_date"),
      travelers: Number(fd.get("travelers")),
      budget: Number(fd.get("budget")),
      currency: fd.get("currency"),
      travel_style: fd.get("travel_style"),
      interests: fd.getAll("interests"),
      accommodation: (fd.get("accommodation") || "").trim(),
      notes: (fd.get("notes") || "").trim()
    };
    if (fd.get("return_date")) payload.return_date = fd.get("return_date");
    else if (fd.get("duration_days")) payload.duration_days = Number(fd.get("duration_days"));
    return payload;
  }

  function clearInvalid() {
    form.querySelectorAll("[aria-invalid]").forEach(function (el) { el.removeAttribute("aria-invalid"); });
  }

  if (discoverLink) {
    discoverLink.addEventListener("click", function () {
      var destination = document.getElementById("destination").value.trim();
      if (destination) {
        var params = new URLSearchParams({ destination: destination, category: "tourist attractions" });
        discoverLink.href = "/discover?" + params.toString();
      } else {
        discoverLink.href = "/discover";
      }
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    clearInvalid();
    lastPayload = readPayload();
    requestPlan(lastPayload);
  });

  // ---------- API ----------
  function postJSON(url, body) {
    var options = { method: "POST", headers: { "Content-Type": "application/json" } };
    if (body !== undefined) options.body = JSON.stringify(body);
    return fetch(url, options).then(function (resp) {
      return resp.json().catch(function () { return {}; }).then(function (data) {
        if (!resp.ok) {
          var err = (data && data.error) || {};
          var e = new Error(err.message || "Something went wrong. Please try again.");
          e.details = err.details || [];
          e.code = err.code || "http_error";
          throw e;
        }
        return data;
      });
    });
  }

  function requestPlan(payload) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Finding options…";
    showLoading();
    workflowId = null;
    fetch("/workflows", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        workflowId = d.workflow_id;
        return postJSON("/workflows/" + encodeURIComponent(workflowId) + "/requirements", payload);
      })
      .then(handleResponse)
      .catch(handleFailure);
  }

  function sendClarification(payload, panel) {
    var send = panel.querySelector(".clarify-row .go");
    var existing = panel.querySelector(".clarify-error");
    if (existing) existing.remove();
    if (send) { send.disabled = true; send.textContent = "Submitting…"; }
    postJSON("/workflows/" + encodeURIComponent(workflowId) + "/clarifications", payload)
      .then(function (body) { handleResponse(body); })
      .catch(function (err) {
        if (send) { send.disabled = false; send.textContent = "Submit answer"; }
        panel.appendChild(h("p", { class: "clarify-error", role: "alert",
          text: err.message || "Please check the values and try again." }));
      });
  }

  function handleFailure(err) {
    stopLoading();
    showError({ message: err.message, details: err.details || [] });
  }

  function handleResponse(body) {
    stopLoading();
    if (!body || body.status === "failed") {
      showError({ message: (body && body.error && body.error.message) || "The conversation ended with an error. Please start again.", details: [] });
      return;
    }
    renderResponse(body);
  }

  var LOADING_STEPS = [
    "Checking your trip details…",
    "Asking the agent to search travel options…",
    "Ranking options against your preferences…"
  ];

  function showLoading() {
    var msg = h("span", { text: LOADING_STEPS[0] });
    results.replaceChildren(
      h("div", { class: "panel loading", role: "status" }, h("div", { class: "spinner", "aria-hidden": "true" }),
        h("div", null, h("strong", { text: "Talking to the travel agent" }), h("br"), msg))
    );
    var i = 0;
    clearInterval(loadingTimer);
    loadingTimer = setInterval(function () { i = Math.min(i + 1, LOADING_STEPS.length - 1); msg.textContent = LOADING_STEPS[i]; }, 3500);
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function stopLoading() { clearInterval(loadingTimer); submitBtn.disabled = false; submitBtn.textContent = "Find travel options"; }

  // ---------- errors ----------
  function showError(err) {
    err = err || {};
    var details = err.details || [];
    details.forEach(function (d) {
      var el = d.field && document.getElementById(d.field);
      if (el) el.setAttribute("aria-invalid", "true");
    });
    var list = details.length ? h("ul", null, details.map(function (d) { return h("li", { text: d.message }); })) : null;
    var retry = h("button", { class: "btn", type: "button", text: "Retry" });
    retry.addEventListener("click", function () { if (lastPayload) requestPlan(lastPayload); });
    var edit = h("button", { class: "btn", type: "button", text: "Edit trip details" });
    edit.addEventListener("click", function () { form.scrollIntoView({ behavior: "smooth" }); document.getElementById("origin").focus(); });
    results.replaceChildren(
      h("div", { class: "panel error", role: "alert" },
        h("h2", { text: "We couldn't finish that request" }),
        h("p", { text: details.length ? "Fix these and try again:" : (err.message || "Something went wrong. Please try again.") }),
        list,
        h("div", { class: "again" }, retry, edit))
    );
    results.focus();
  }

  // ---------- rendering ----------
  function badge(text, cls) { return h("span", { class: "badge " + cls, text: text }); }

  var STATUS_LABELS = {
    collecting_requirements: "Collecting details",
    awaiting_clarification: "Needs more details",
    requirements_complete: "Requirements complete",
    failed: "Failed"
  };

  function renderResponse(body) {
    var frag = document.createDocumentFragment();
    if (body.requirements) frag.appendChild(summaryPanel(body));

    if (body.clarification_questions && body.clarification_questions.length) {
      frag.appendChild(clarificationPanel(body.clarification_questions));
    } else if (body.recommendation) {
      frag.appendChild(recommendationPanel(body.recommendation, body.requirements));
    }

    var again = h("button", { class: "btn", type: "button", text: "Plan another trip" });
    again.addEventListener("click", function () { workflowId = null; form.scrollIntoView({ behavior: "smooth" }); });
    frag.appendChild(h("div", { class: "again" }, again));

    results.replaceChildren(frag);
    results.scrollIntoView({ behavior: "smooth", block: "start" });
    results.focus({ preventScroll: true });
  }

  function summaryPanel(body) {
    var req = body.requirements;
    var trip = req.trip || {};
    var dates = trip.dates || null;
    var when = "Dates not set";
    if (dates && dates.date_range) when = prettyDate(dates.date_range.start) + " to " + prettyDate(dates.date_range.end);
    else if (dates && dates.duration_days) when = dates.duration_days + (dates.duration_days === 1 ? " day" : " days");

    var travelers = req.travelers && req.travelers.count;
    var budget = req.budget || {};
    var facts = [when];
    if (travelers) facts.push(travelers + (travelers === 1 ? " traveler" : " travelers"));
    if (budget.total != null) facts.push("budget " + money(Number(budget.total), budget.currency) + (budget.is_approximate ? " (target)" : ""));

    var status = body.status || "";
    var badges = h("div", { class: "badges" },
      badge(STATUS_LABELS[status] || titleCase(status), status === "requirements_complete" ? "live" : "neutral"));

    return h("div", { class: "panel" },
      h("div", { class: "summary-head" },
        h("div", null,
          h("h2", { text: (trip.origin || "Origin") + " to " + (trip.destination || "Destination") }),
          h("p", { class: "facts", text: facts.join(" · ") }))),
      badges,
      body.reply ? h("p", { class: "fine", text: body.reply }) : null);
  }

  // Deterministic clarification inputs, keyed by the backend clarification
  // field. Answers are merged server-side without involving an LLM, so they
  // work with the credential-free default (stub) provider.
  var CLARIFY_INPUTS = {
    "trip.origin": [{ key: "origin", label: "From", type: "text", placeholder: "e.g. Pune", autocomplete: "off", maxlength: "80" }],
    "trip.destination": [{ key: "destination", label: "To", type: "text", placeholder: "e.g. Goa", autocomplete: "off", maxlength: "80" }],
    "travel.timing": [
      { key: "departure_date", label: "Departure date", type: "date" },
      { key: "duration_days", label: "Trip length (days)", type: "number", min: "1", max: "60", inputmode: "numeric" },
      { key: "return_date", label: "Return date", type: "date" }
    ],
    "budget.total": [{ key: "budget", label: "Target budget", type: "number", min: "1", step: "any", inputmode: "decimal" }],
    "travelers.count": [{ key: "travelers", label: "Travelers", type: "number", min: "1", max: "20", inputmode: "numeric" }]
  };
  var NUMERIC_ANSWER_FIELDS = { duration_days: true, travelers: true, budget: true };

  function clarificationPanel(questions) {
    var specs = [];
    var seen = {};
    var unresolved = [];
    questions.forEach(function (q) {
      var list = CLARIFY_INPUTS[q.field];
      if (!list) { unresolved.push(q.field); return; }
      list.forEach(function (spec) { if (!seen[spec.key]) { seen[spec.key] = true; specs.push(spec); } });
    });

    var inputs = {};
    var grid = specs.map(function (spec) {
      var id = "clarify-" + spec.key;
      var attrs = { id: id, name: spec.key, type: spec.type };
      ["placeholder", "min", "max", "step", "inputmode", "maxlength", "autocomplete"].forEach(function (k) {
        if (spec[k] != null) attrs[k] = spec[k];
      });
      var input = h("input", attrs);
      if (lastPayload && lastPayload[spec.key] != null && lastPayload[spec.key] !== "") {
        input.value = lastPayload[spec.key];
      }
      inputs[spec.key] = input;
      return h("div", { class: "field" }, h("label", { for: id, text: spec.label }), input);
    });

    var send = h("button", { class: "go", type: "button", text: "Submit answer" });
    var panel = h("div", { class: "panel" },
      h("h2", { text: "A few details are still missing" }),
      h("ul", { class: "plain" }, questions.map(function (q) { return h("li", { text: q.question }); })),
      grid.length ? h("div", { class: "clarify-grid" }, grid) : null,
      grid.length ? h("div", { class: "clarify-row" }, send) : null,
      !grid.length ? h("p", { class: "fine", text: "We can't fill this in automatically. Please update the trip details in the form above and submit again." }) : null,
      unresolved.length ? h("p", { class: "fine", text: "Still needed: " + unresolved.join(", ") + "." }) : null);

    if (grid.length) {
      send.addEventListener("click", function () {
        var payload = {};
        specs.forEach(function (spec) {
          var el = inputs[spec.key];
          var raw = (el.value || "").trim();
          el.removeAttribute("aria-invalid");
          if (!raw) return;
          var value = NUMERIC_ANSWER_FIELDS[spec.key] ? Number(raw) : raw;
          if (typeof value === "number" && isNaN(value)) { el.setAttribute("aria-invalid", "true"); return; }
          payload[spec.key] = value;
        });
        if (!Object.keys(payload).length) {
          inputs[specs[0].key].setAttribute("aria-invalid", "true");
          inputs[specs[0].key].focus();
          return;
        }
        sendClarification(payload, panel);
      });
    }
    return panel;
  }

  function recommendationPanel(rec, req) {
    var panel = h("div", { class: "panel" });
    var titleByStatus = {
      ok: "Recommended options",
      no_results: "No options found",
      no_feasible_options: "No options match your hard requirements",
      provider_unavailable: "Travel search is unavailable",
      decision_error: "We couldn't evaluate the options"
    };
    panel.appendChild(h("h2", { text: titleByStatus[rec.status] || "Recommendation" }));

    if (rec.status === "ok" && rec.options && rec.options.length) {
      panel.appendChild(h("p", { class: "notice" },
        h("strong", { text: "Demo data, not bookable. " }),
        h("span", { text: "These options are synthetic fixtures from the demo search provider. Prices are not live and nothing here can be booked." })));
      panel.appendChild(h("div", { class: "options" }, rec.options.map(function (opt) { return optionCard(opt, req); })));
    } else if (rec.status === "provider_unavailable") {
      panel.appendChild(h("p", { text: "The travel search provider could not be reached. This is a provider failure, not an empty result. Please try again later." }));
    } else if (rec.status === "decision_error") {
      panel.appendChild(h("p", { text: "The options were retrieved but could not be evaluated. No recommendation was produced." }));
    } else if (rec.status === "no_results") {
      panel.appendChild(h("p", { text: "The search provider returned no options for this route." }));
    } else if (rec.status === "no_feasible_options") {
      panel.appendChild(h("p", { text: "Every option was rejected by your hard requirements." }));
    }

    if (rec.rejected && rec.rejected.length) {
      panel.appendChild(h("h3", { text: "Rejected options" }));
      panel.appendChild(h("ul", { class: "plain" }, rec.rejected.slice(0, 8).map(function (r) {
        return h("li", null,
          h("strong", { text: r.candidate && r.candidate.label ? r.candidate.label : "Option" }),
          " — ",
          h("span", { text: (r.reasons || []).map(titleCase).join(", ") }));
      })));
    }
    return panel;
  }

  function optionCard(opt, req) {
    var c = opt.candidate || {};
    var currency = c.currency || (req && req.budget && req.budget.currency) || "INR";
    var badges = h("div", { class: "badges" });
    if (c.bookable) badges.appendChild(badge("Bookable", "live"));
    else badges.appendChild(badge("Demo · not bookable", "demo"));
    if (c.source) badges.appendChild(badge("Source: " + c.source, c.source === "verified" ? "live" : "neutral"));

    var meta = [];
    if (c.transport_mode) meta.push(titleCase(c.transport_mode));
    if (c.duration_days != null) meta.push(c.duration_days + " day(s)");
    if (c.departure_time) meta.push("departs " + String(c.departure_time).replace(/_/g, " "));

    var priceText = c.price != null ? money(Number(c.price), currency) : "Price unavailable";
    var score = typeof opt.score === "number" ? Math.round(opt.score * 100) + "% match" : null;

    return h("article", { class: "option" },
      h("div", { class: "option-head" },
        h("div", null,
          h("span", { class: "rank", text: "#" + opt.rank }),
          h("h3", { text: c.label || "Option" }),
          meta.length ? h("p", { class: "facts", text: meta.join(" · ") }) : null),
        h("div", { class: "option-price" },
          h("div", { class: "v", text: priceText }),
          score ? h("div", { class: "fine", text: score }) : null)),
      badges,
      opt.reasons && opt.reasons.length
        ? h("ul", { class: "plain reasons" }, opt.reasons.map(function (r) { return h("li", { text: r }); }))
        : null,
      h("p", { class: "fine", text: "Booking is not available in this build." }));
  }
})();
