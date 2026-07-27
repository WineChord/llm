function revealCodeDisclosureTarget() {
  if (!location.hash) return;
  let id = location.hash.slice(1);
  try {
    id = decodeURIComponent(id);
  } catch (_) {
    return;
  }
  const target = document.getElementById(id);
  if (!target) return;
  for (
    let node = target.closest("details");
    node;
    node = node.parentElement?.closest("details")
  ) {
    node.open = true;
  }
}

function enableCodeDisclosureKeyboard() {
  document
    .querySelectorAll("details.code-disclosure > summary")
    .forEach((summary) => {
      if (summary.dataset.codeDisclosureKeyboard === "true") return;
      summary.dataset.codeDisclosureKeyboard = "true";
      summary.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") {
          return;
        }
        event.preventDefault();
        summary.parentElement.open = !summary.parentElement.open;
      });
    });
}

window.addEventListener("hashchange", revealCodeDisclosureTarget);
document$.subscribe(() => {
  enableCodeDisclosureKeyboard();
  requestAnimationFrame(revealCodeDisclosureTarget);
});
