import csv
import json
from pathlib import Path

CSV_DIR     = Path("./pokeapi/data/v2/csv")
OUTPUT_FILE = "datos.jsonl"
GEN_MAX     = 1  # hasta la primera generación
LANG_ES     = 7  # id del español en la base de datos


def cargar_csv(nombre):
    rows = []
    with open(CSV_DIR / nombre, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

print("Cargando CSVs...")

pokemon_rows        = cargar_csv("pokemon.csv")
species_rows        = cargar_csv("pokemon_species.csv")
species_names_rows  = cargar_csv("pokemon_species_names.csv")
types_rows          = cargar_csv("types.csv")
type_names_rows     = cargar_csv("type_names.csv")
pokemon_types_rows  = cargar_csv("pokemon_types.csv")
stats_rows          = cargar_csv("stats.csv")
pokemon_stats_rows  = cargar_csv("pokemon_stats.csv")
abilities_rows      = cargar_csv("abilities.csv")
ability_names_rows  = cargar_csv("ability_names.csv")
pokemon_abilities_rows = cargar_csv("pokemon_abilities.csv")
flavor_rows         = cargar_csv("pokemon_species_flavor_text.csv")
genera_rows         = cargar_csv("pokemon_species_names.csv")


# Tipos en español
tipo_nombre = {}
for r in type_names_rows:
    if r["local_language_id"] == str(LANG_ES):
        tipo_nombre[r["type_id"]] = r["name"]

# Habilidades en español
hab_nombre = {}
for r in ability_names_rows:
    if r["local_language_id"] == str(LANG_ES):
        hab_nombre[r["ability_id"]] = r["name"]

# Nombres de pokemon en español
poke_nombre = {}
poke_genero = {}
for r in species_names_rows:
    if r["local_language_id"] == str(LANG_ES):
        poke_nombre[r["pokemon_species_id"]] = r["name"]
        poke_genero[r["pokemon_species_id"]] = r["genus"]

# Descripción en español (primera disponible por pokemon)
poke_desc = {}
for r in flavor_rows:
    if r["language_id"] == str(LANG_ES):
        sid = r["species_id"]
        if sid not in poke_desc:
            texto = r["flavor_text"].replace("\n", " ").replace("\f", " ")
            poke_desc[sid] = " ".join(texto.split())

# Tipos por pokemon
poke_tipos = {}
for r in pokemon_types_rows:
    pid = r["pokemon_id"]
    if pid not in poke_tipos:
        poke_tipos[pid] = []
    poke_tipos[pid].append((r["slot"], r["type_id"]))

# Stats por pokemon
stat_nombre = {"1": "PS", "2": "Ataque", "3": "Defensa",
               "4": "Ataque Especial", "5": "Defensa Especial", "6": "Velocidad"}
poke_stats = {}
for r in pokemon_stats_rows:
    pid = r["pokemon_id"]
    if pid not in poke_stats:
        poke_stats[pid] = {}
    sname = stat_nombre.get(r["stat_id"], r["stat_id"])
    poke_stats[pid][sname] = r["base_stat"]

# Habilidades por pokemon
poke_habs = {}
for r in pokemon_abilities_rows:
    pid = r["pokemon_id"]
    if pid not in poke_habs:
        poke_habs[pid] = {"normales": [], "oculta": None}
    nombre_hab = hab_nombre.get(r["ability_id"], r["ability_id"])
    if r["is_hidden"] == "1":
        poke_habs[pid]["oculta"] = nombre_hab
    else:
        poke_habs[pid]["normales"].append(nombre_hab)

# Peso y altura por pokemon
poke_info = {}
for r in pokemon_rows:
    poke_info[r["id"]] = {
        "altura": int(r["height"]) / 10,
        "peso":   int(r["weight"]) / 10,
    }

# Generación por species
species_gen = {}
species_legendario = {}
species_mitico = {}
species_captura = {}
for r in species_rows:
    species_gen[r["id"]]        = r["generation_id"]
    species_legendario[r["id"]] = r["is_legendary"]
    species_mitico[r["id"]]     = r["is_mythical"]
    species_captura[r["id"]]    = r["capture_rate"]

gens = {
    "1": "primera", "2": "segunda", "3": "tercera", "4": "cuarta",
    "5": "quinta",  "6": "sexta",   "7": "séptima", "8": "octava"
}

# GENERAR PARES
print("Generando pares pregunta-respuesta...")

pares = []
total_pokemon = 0

for r in pokemon_rows:
    pid      = r["id"]
    sid      = r["species_id"]

    # Filtrar por generación
    gen = species_gen.get(sid, "99")
    if int(gen) > GEN_MAX:
        continue

    # Necesitamos nombre en español
    nombre = poke_nombre.get(sid)
    if not nombre:
        continue

    total_pokemon += 1

    # --- TIPOS ---
    tipos_raw = sorted(poke_tipos.get(pid, []), key=lambda x: x[0])
    tipos = [tipo_nombre.get(t[1], t[1]) for t in tipos_raw]
    tipo_str = "/".join(tipos)

    if tipo_str:
        pares.append({"question": f"¿De qué tipo es {nombre}?",
                      "answer": f"{nombre} es de tipo {tipo_str}."})
        pares.append({"question": f"¿Cuál es el tipo de {nombre}?",
                      "answer": f"El tipo de {nombre} es {tipo_str}."})
        if len(tipos) == 2:
            pares.append({"question": f"¿{nombre} tiene doble tipo?",
                          "answer": f"Sí, {nombre} tiene doble tipo: {tipos[0]} y {tipos[1]}."})
        else:
            pares.append({"question": f"¿{nombre} tiene doble tipo?",
                          "answer": f"No, {nombre} solo tiene un tipo: {tipos[0]}."})

    # --- NÚMERO ---
    pares.append({"question": f"¿Cuál es el número de {nombre} en la Pokédex?",
                  "answer": f"{nombre} es el número {pid} en la Pokédex."})

    # --- ESTADÍSTICAS ---
    stats = poke_stats.get(pid, {})
    if stats:
        stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
        pares.append({"question": f"¿Cuáles son las estadísticas base de {nombre}?",
                      "answer": f"Las estadísticas base de {nombre} son: {stats_str}."})
        if "PS" in stats:
            pares.append({"question": f"¿Cuántos PS base tiene {nombre}?",
                          "answer": f"{nombre} tiene {stats['PS']} PS base."})
        if "Velocidad" in stats:
            pares.append({"question": f"¿Cuál es la velocidad base de {nombre}?",
                          "answer": f"La velocidad base de {nombre} es {stats['Velocidad']}."})
        if "Ataque" in stats:
            pares.append({"question": f"¿Cuál es el ataque base de {nombre}?",
                          "answer": f"El ataque base de {nombre} es {stats['Ataque']}."})
        if "Defensa" in stats:
            pares.append({"question": f"¿Cuál es la defensa base de {nombre}?",
                          "answer": f"La defensa base de {nombre} es {stats['Defensa']}."})

    # --- PESO Y ALTURA ---
    info = poke_info.get(pid, {})
    if info:
        pares.append({"question": f"¿Cuánto pesa {nombre}?",
                      "answer": f"{nombre} pesa {info['peso']} kg."})
        pares.append({"question": f"¿Cuánto mide {nombre}?",
                      "answer": f"{nombre} mide {info['altura']} metros."})

    # --- HABILIDADES ---
    habs = poke_habs.get(pid, {})
    normales = habs.get("normales", [])
    oculta   = habs.get("oculta")
    if normales:
        hab_str = " y ".join(normales)
        pares.append({"question": f"¿Cuál es la habilidad de {nombre}?",
                      "answer": f"La habilidad de {nombre} es {hab_str}."})
        pares.append({"question": f"¿Qué habilidades tiene {nombre}?",
                      "answer": f"{nombre} tiene las siguientes habilidades: {hab_str}."})
    if oculta:
        pares.append({"question": f"¿Cuál es la habilidad oculta de {nombre}?",
                      "answer": f"La habilidad oculta de {nombre} es {oculta}."})

    # --- DESCRIPCIÓN ---
    desc = poke_desc.get(sid)
    if desc:
        pares.append({"question": f"¿Cómo se describe a {nombre} en la Pokédex?",
                      "answer": f"Según la Pokédex: {desc}"})
        pares.append({"question": f"¿Quién es {nombre}?",
                      "answer": f"{nombre} es un Pokémon de tipo {tipo_str}. {desc}"})

    # --- GÉNERO / CATEGORÍA ---
    genero = poke_genero.get(sid)
    if genero:
        pares.append({"question": f"¿A qué categoría pertenece {nombre}?",
                      "answer": f"{nombre} pertenece a la categoría {genero}."})

    # --- TASA DE CAPTURA ---
    captura = species_captura.get(sid)
    if captura:
        pares.append({"question": f"¿Cuál es la tasa de captura de {nombre}?",
                      "answer": f"La tasa de captura de {nombre} es {captura} sobre 255."})

    # --- GENERACIÓN ---
    if gen in gens:
        pares.append({"question": f"¿De qué generación es {nombre}?",
                      "answer": f"{nombre} es de la {gens[gen]} generación."})

    # --- LEGENDARIO / MÍTICO ---
    if species_legendario.get(sid) == "1":
        pares.append({"question": f"¿Es {nombre} un Pokémon legendario?",
                      "answer": f"Sí, {nombre} es un Pokémon legendario."})
    elif species_mitico.get(sid) == "1":
        pares.append({"question": f"¿Es {nombre} un Pokémon mítico?",
                      "answer": f"Sí, {nombre} es un Pokémon mítico."})
    else:
        pares.append({"question": f"¿Es {nombre} un Pokémon legendario?",
                      "answer": f"No, {nombre} no es un Pokémon legendario."})

# GUARDAR
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for par in pares:
        f.write(json.dumps(par, ensure_ascii=False) + "\n")

print("=" * 50)
print(f"Pokémon procesados : {total_pokemon}")
print(f"Pares generados    : {len(pares)}")
print(f"Archivo guardado   : {OUTPUT_FILE}")
print("Ahora ejecuta      : python3 entrenar.py")
print("=" * 50)
