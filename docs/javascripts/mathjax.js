window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    typeset: false
  }
};

let previousMathTargets = [];
let pendingMathTypeset = Promise.resolve();

document$.subscribe(() => {
  pendingMathTypeset = pendingMathTypeset
    .then(() => {
      if (previousMathTargets.length) {
        MathJax.typesetClear(previousMathTargets);
      }
      MathJax.texReset();
      previousMathTargets = [...document.querySelectorAll(".arithmatex")]
        .filter((element) =>
          element.querySelector(":scope > mjx-container") === null
        );
      return previousMathTargets.length
        ? MathJax.typesetPromise(previousMathTargets)
        : undefined;
    })
    .catch((error) => console.error("MathJax typesetting failed", error));
});
