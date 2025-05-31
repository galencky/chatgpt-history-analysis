from collections import defaultdict
import pprint

def survey_conversation_keys(conversations, print_progress=True):
    """
    Survey the structure of your conversations.json file.
    Returns: survey (dict with conversation/mapping_node/message keys)
    """
    survey = {
        "conversation": defaultdict(set),
        "mapping_node": defaultdict(set),
        "message": defaultdict(set),
    }

    if print_progress:
        print("🔎 Surveying schema of conversations...")

    # Traverse all conversations and collect key types
    for idx, conv in enumerate(conversations):
        if print_progress and ((idx + 1) % 100 == 0 or idx == len(conversations) - 1):
            print(f"  ...processed {idx + 1} / {len(conversations)} conversations")

        # --- Conversation-level keys ---
        for k, v in conv.items():
            survey["conversation"][k].add(type(v).__name__)

        mapping = conv.get("mapping", {})
        if not isinstance(mapping, dict):
            continue

        # --- Mapping-node-level keys ---
        for node in mapping.values():
            for k, v in node.items():
                survey["mapping_node"][k].add(type(v).__name__)

            # --- Message-level keys ---
            message = node.get("message")
            if isinstance(message, dict):
                for k, v in message.items():
                    survey["message"][k].add(type(v).__name__)

    if print_progress:
        print("\n✅ Finished schema survey.\n")

        # Display key findings
        print("🔑 Conversation-level keys + types:")
        pprint.pprint(dict(survey["conversation"]))

        print("\n🔧 Mapping-node-level keys + types:")
        pprint.pprint(dict(survey["mapping_node"]))

        print("\n🗨️ Message-level keys + types:")
        pprint.pprint(dict(survey["message"]))

    return survey
