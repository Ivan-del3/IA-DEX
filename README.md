# IA Dex — Entrena tu propia IA local

> Guía paso a paso para entrenar un modelo de lenguaje con tus propios datos y desplegarlo como chatbot web. Sin APIs de pago, sin conexión a internet, todo en local.

![Fine-tuning](https://img.shields.io/badge/técnica-Fine--tuning%20LoRA-cc0000?style=flat-square)
![Ollama](https://img.shields.io/badge/motor-Ollama-111111?style=flat-square)
![Modelo](https://img.shields.io/badge/modelo-Qwen2.5--1.5B-white?style=flat-square)
![Stack](https://img.shields.io/badge/stack-HTML%20·%20CSS%20·%20JS-cc0000?style=flat-square)

---

## ¿Qué vas a aprender?

Esta guía explica cómo entrenar un modelo de lenguaje con datos propios usando **fine-tuning con LoRA**, convertirlo al formato que usa Ollama y desplegarlo en un chat web. Como ejemplo práctico se ha entrenado un asistente especializado en los 151 Pokémon de la primera generación.

---

## Nota sobre el hardware

El fine-tuning se hace de forma óptima con una **GPU NVIDIA**. Las GPUs NVIDIA tienen soporte para **CUDA**, una tecnología que permite ejecutar miles de operaciones matemáticas en paralelo, reduciendo el tiempo de entrenamiento de horas a minutos.

Sin GPU el entrenamiento recae sobre la CPU, que realiza esas mismas operaciones de forma secuencial y es mucho más lenta. Por eso en este proyecto se ha entrenado **solo con la primera generación** (151 Pokémon, ~2.800 ejemplos): es la cantidad que permite completar el entrenamiento en un tiempo razonable con CPU (~8 horas). Con una GPU NVIDIA se podrían usar los 898 Pokémon existentes en menos de 30 minutos.

---

## Requisitos

- Linux o WSL2 (Windows Subsystem for Linux)
- Python 3.10+
- [Ollama](https://ollama.com) instalado
- 16GB de RAM mínimo
- GPU NVIDIA recomendada

---

## Flujo del proyecto

```
Datos CSV (PokeAPI)
        ↓
extraer_csv.py  →  datos.jsonl
        ↓
entrenar.py  →  adaptador LoRA
        ↓
fusionar.py  →  modelo completo
        ↓
convert_hf_to_gguf.py  →  modelo.gguf
        ↓
ollama create  →  modelo en Ollama
        ↓
Chat web (HTML + JS)
```

---

## Paso 1 — Instalar dependencias

```bash
pip install torch transformers peft trl datasets --break-system-packages
```

Clona **llama.cpp**, necesario para convertir el modelo al formato GGUF que usa Ollama:

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt --break-system-packages
```

---

## Paso 2 — Preparar los datos

El modelo aprende a partir de pares pregunta-respuesta en formato JSONL, una entrada por línea:

```json
{"question": "¿De qué tipo es Pikachu?", "answer": "Pikachu es de tipo Eléctrico."}
{"question": "¿Cuál es la habilidad de Charizard?", "answer": "La habilidad de Charizard es Mar Llamas."}
```

En este proyecto los datos se extraen del repositorio oficial de PokeAPI. Primero clona el repo:

```bash
git clone https://github.com/PokeAPI/pokeapi.git
```

Luego ejecuta el script de extracción:

```bash
python3 extraer_csv.py
```

El script lee los archivos CSV de PokeAPI y genera automáticamente entre 12 y 16 preguntas por Pokémon cubriendo tipo, estadísticas, habilidades, peso, altura, descripción, generación y si es legendario. Para los 151 Pokémon de primera generación genera **~2.800 pares**.

Comprueba cuántos ejemplos se han generado:

```bash
wc -l datos.jsonl
```

> **-  ¿Quieres entrenar con tus propios datos?** Simplemente crea tu archivo `datos.jsonl` con el mismo formato. El modelo aprenderá de lo que tú le pongas, no hace falta que sean Pokémon.

---

## Paso 3 — Entrenar el modelo

El script `entrenar.py` descarga el modelo base de HuggingFace y aplica **LoRA (Low-Rank Adaptation)**, una técnica que en lugar de modificar todos los pesos del modelo añade unas matrices de adaptación pequeñas encima. Esto hace el proceso mucho más eficiente en memoria y tiempo.

Lanza el entrenamiento en segundo plano para que no se corte si cierras la terminal:

```bash
nohup python3 entrenar.py > entrenamiento.log 2>&1 &
```

Sigue el progreso en cualquier momento:

```bash
tail -f entrenamiento.log
```

Los parámetros más importantes del entrenamiento:

```python
MODEL_NAME            = "Qwen/Qwen2.5-1.5B"  # modelo base
NUM_TRAIN_EPOCHS      = 5      # veces que repasa todos los datos
LEARNING_RATE         = 2e-4   # velocidad de aprendizaje
r                     = 16     # tamaño del adaptador LoRA
lora_alpha            = 32     # peso del adaptador sobre el modelo
```

Cuando termine verás en el log:

```
ENTRENAMIENTO COMPLETADO
Adaptador guardado en: pokemon-qwen-lora
```

Los resultados del entrenamiento en este proyecto:

| Parámetro | Valor |
|---|---|
| Epochs | 5 |
| Loss inicial | 2.26 |
| Loss final | 0.37 |
| Duración en CPU | ~8 horas |

La **loss** mide el error del modelo. Que baje de 2.26 a 0.37 indica que el modelo aprendió correctamente los datos.

---

## Paso 4 — Fusionar el adaptador

El entrenamiento genera un adaptador LoRA, no un modelo completo. Hay que fusionarlo con el modelo base:

```bash
python3 fusionar.py
```

Esto carga el modelo base, aplica el adaptador y guarda el resultado como un modelo completo listo para convertir.

---

## Paso 5 — Convertir a GGUF e importar a Ollama

Ollama usa el formato **GGUF**, que comprime el modelo y lo optimiza para inferencia en CPU con cuantización Q8_0:

```bash
python3 llama.cpp/convert_hf_to_gguf.py \
    ./pokemon-qwen-fusionado \
    --outfile ./pokemon-qwen.gguf \
    --outtype q8_0
```

Crea el `Modelfile` que define el comportamiento del modelo en Ollama:

```
FROM ./pokemon-qwen.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
PARAMETER stop "### Pregunta:"
SYSTEM """
Eres una Pokédex entrenada. Responde SOLO en español.
Responde únicamente con la información que conoces de tu entrenamiento.
Si no tienes la respuesta, dilo claramente.
"""
```

Importa el modelo:

```bash
ollama create pokemon-ia -f Modelfile
```

Prueba que funciona:

```bash
ollama run pokemon-ia "¿De qué tipo es Pikachu?"
```

---

## Paso 6 — Desplegar el chat web

El chat es una página HTML estática que se comunica con Ollama mediante su API REST. Las respuestas se muestran en streaming mientras el modelo las genera.

La parte clave del JavaScript es el prompt, que usa el mismo formato con el que se entrenó el modelo:

```javascript
async function enviar() {
  const pregunta = input.value.trim();

  const response = await fetch("http://TU_IP:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "pokemon-ia",
      prompt: `### Pregunta:\n${pregunta}\n\n### Respuesta:\n`,
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const data = JSON.parse(decoder.decode(value));
    if (data.response) output.textContent += data.response;
  }
}
```

> **- Importante:** El prompt usa `### Pregunta / ### Respuesta` porque es exactamente el formato con el que se entrenó el modelo. Usar el mismo formato en inferencia mejora la precisión de las respuestas.

Para que Ollama acepte peticiones desde el navegador configura CORS:

```bash
sudo systemctl edit ollama
```

```ini
[Service]
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_HOST=0.0.0.0"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

## Estructura del proyecto

```
proyecto/
├── datos/
│   ├── extraer_csv.py     # Genera datos.jsonl desde PokeAPI
│   └── datos.jsonl        # Pares pregunta-respuesta generados
│
├── entrenamiento/
│   └── entrenar.py        # Fine-tuning con LoRA
│
├── conversion/
│   ├── fusionar.py        # Une adaptador LoRA + modelo base
│   └── convertir_gguf.sh  # Convierte a GGUF e importa a Ollama
│
├── modelo/
│   ├── modelo.gguf        # Modelo final (~1.6GB)
│   └── Modelfile          # Configuración para Ollama
│
└── web/
    ├── index.html
    ├── styles.css
    └── script.js
```

---

## Resultado

```
> ¿De qué tipo es Pikachu?
  Pikachu es de tipo Eléctrico.

> ¿Cuál es la habilidad oculta de Charizard?
  La habilidad oculta de Charizard es Mar Llamas.

> ¿Cuánto pesa Snorlax?
  Snorlax pesa 460.0 kg.
```

---
