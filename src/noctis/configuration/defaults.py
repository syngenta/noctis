package_name = "noctis"

x = {
    "settings": {"NEO4J_DEV_URL": "bolt://localhost:7687"},
    "secrets": {
        "NEO4J_DEV_PASSWORD": "<put the correct value here and keep it secret>",
        "NEO4J_DEV_USER": "<put the correct value here and keep it secret>",
    },
    "schema": {
        "NODES": {
            "node_molecule": "Molecule",
            "node_chemequation": "ChemicalEquation",
        },
        "RELATIONSHIPS": {
            "relationship_reactant": "REACTANT",
            "relationship_product": "PRODUCT",
        },
        "START_END_NODES": {
            "relationship_reactant": ["node_molecule", "node_chemequation"],
            "relationship_product": ["node_chemequation", "node_molecule"],
        },
    },
}

settings = {key: value for key, value in x.get("settings").items()}
schema = {key: value for key, value in x.get("schema").items()}
secrets = {key: value for key, value in x.get("secrets").items()}