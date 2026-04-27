const OLLAMA_URL = "http://172.19.14.142:11434/api/generate";
const MODELO = "pokemon-ia";

const chat = document.getElementById("chat");
const input = document.getElementById("input");
const btn = document.getElementById("btn");

input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !btn.disabled) enviar();
});

function agregarMensaje(texto, tipo) {
  const div = document.createElement("div");
  div.className = "msg " + tipo;
  div.textContent = texto;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function enviar() {
  const pregunta = input.value.trim();
  if (!pregunta) return;
  input.value = "";
  btn.disabled = true;
  input.disabled = true;

  agregarMensaje(pregunta, "user");
  const botDiv = agregarMensaje("Pensando...", "bot cargando");

  try {
    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODELO,
        prompt: `### Pregunta:\n${pregunta}\n\n### Respuesta:\n`,
        stream: true
      })
    });

    botDiv.className = "msg bot";
    botDiv.textContent = "";

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const line of decoder.decode(value).split("\n").filter(l => l.trim())) {
        try {
          const data = JSON.parse(line);
          if (data.response) {
            botDiv.textContent += data.response;
            chat.scrollTop = chat.scrollHeight;
          }
        } catch {}
      }
    }
  } catch {
    botDiv.className = "msg bot";
    botDiv.textContent = "Error al conectar con Ollama.";
  }

  btn.disabled = false;
  input.disabled = false;
  input.focus();
}