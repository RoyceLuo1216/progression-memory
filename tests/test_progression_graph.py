import pytest

from progression_memory.models.progression import Progression
from progression_memory.storage.progression_graph import (
    GraphRelation,
    ProgressionGraph,
)


def test_progression_graph_supports_multi_parent_and_rejects_cycles() -> None:
    progressions = {
        "health": Progression(id="health", title="Health", scope="health"),
        "running": Progression(id="running", title="Running", scope="running"),
        "return": Progression(id="return", title="Return", scope="return to running"),
    }
    graph = ProgressionGraph(progressions)

    graph.link_progressions("health", "return")
    graph.link_progressions("running", "return")

    assert set(graph.parents("return", GraphRelation.PARENT_CHILD)) == {"health", "running"}
    assert set(progressions["return"].parent_ids) == {"health", "running"}
    assert "return" in progressions["health"].child_ids
    assert "return" in progressions["running"].child_ids
    assert graph.validate()

    with pytest.raises(ValueError, match="cycle"):
        graph.link_progressions("return", "health")


def test_progression_graph_contains_segments_and_topologically_orders_nodes() -> None:
    progression = Progression(id="p", title="P", scope="test")
    graph = ProgressionGraph({"p": progression})
    graph.add_segment("s1", "p")
    graph.add_segment("s2", "p")

    assert set(graph.children("p", GraphRelation.CONTAINS)) == {"s1", "s2"}
    assert set(graph.ancestors("s1")) == {"p"}
    order = graph.topological_order()
    assert order.index("p") < order.index("s1")
    assert order.index("p") < order.index("s2")
