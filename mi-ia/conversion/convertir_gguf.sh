#!/bin/bash
set -e

MODELO_FUSIONADO="./pokemon-qwen-fusionado"
MODELO_GGUF="./pokemon-qwen.gguf"
NOMBRE_OLLAMA="pokemon-ia"
LLAMA_CPP_DIR="./llama.cpp"


if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    pip install -r requirements.txt --break-system-packages
    cd ..
else
    echo "llama.cpp ya existe, saltando..."
fi

python3 llama.cpp/convert_hf_to_gguf.py \
    "$MODELO_FUSIONADO" \
    --outfile "$MODELO_GGUF" \
    --outtype q8_0


cat > Modelfile.final << EOF
FROM $MODELO_GGUF
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
SYSTEM """
Eres una Pokédex entrenada. Responde SOLO en español.
Responde únicamente con la información que conoces de tu entrenamiento.
Si no tienes la respuesta, dilo claramente.
"""
EOF

ollama rm $NOMBRE_OLLAMA 2>/dev/null || true
ollama create $NOMBRE_OLLAMA -f Modelfile.final

echo ""
echo "=================================================="
echo "COMPLETADO"
echo "Prueba: ollama run $NOMBRE_OLLAMA '¿Qué tipo es Pikachu?'"
echo "=================================================="
