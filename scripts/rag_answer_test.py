from src.rag.retriever import Retriever
from src.rag.generator import AnswerGenerator


def main():
    query = "What is starcraft 2? Who is Kerrigan and how is she connected to the Zerg hive?\n"

    retriever = Retriever()
    generator = AnswerGenerator()

    r = retriever.retrieve(query)  # uses top_k from config
    answer = generator.generate(query, r["retrieved_context"])

    print("\nANSWER:\n")
    print(answer)

    # print("\nCITATIONS (retrieved):\n")
    # for c in r["citations"]:
    #     print(f"- [{c['id']}] {c['title']}")

    # used = [c for c in r["citations"] if f"[{c['id']}]" in answer]
    # print("\nCITATIONS (referenced in answer):\n")
    # for c in used:
    #     print(f"- [{c['id']}] {c['title']}")
    #
    # print("\nRERANK DEBUG (top candidates):")
    # for d in r["ranked"][:10]:
    #     print(f"[{d.doc_id}] rerank={d.rerank_score:.4f} dist={d.distance:.4f} | {d.title}")


if __name__ == "__main__":
    main()
