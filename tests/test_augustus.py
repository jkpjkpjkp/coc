import pytest
from coc.tree.augustus import AugustusTreeSearch
from coc.tool.vqa.gemini import Gemini
from coc.exec.mod import Exec
from coc.tool.context.mod import Context

def test_augustus_initialization():
    """Test that AugustusTreeSearch can be initialized with required components."""
    vlm = Gemini()
    exec_engine = Exec()
    context = Context()
    
    tree = AugustusTreeSearch(vlm=vlm, exec_engine=exec_engine, context=context)
    assert tree.vlm == vlm
    assert tree.exec_engine == exec_engine
    assert tree.context == context

def test_single_rollout():
    """Test that a single rollout can be executed with basic code generation."""
    vlm = Gemini()
    exec_engine = Exec()
    context = Context()
    
    tree = AugustusTreeSearch(vlm=vlm, exec_engine=exec_engine, context=context)
    result = tree.single_rollout("Write code to print 'Hello World'")
    assert result is not None
    assert isinstance(result, dict)
    assert "code" in result
    assert "output" in result
    assert "success" in result

def test_tree_search():
    """Test that the tree search can explore multiple paths."""
    vlm = Gemini()
    exec_engine = Exec()
    context = Context()
    
    tree = AugustusTreeSearch(vlm=vlm, exec_engine=exec_engine, context=context)
    result = tree.search("Write code to calculate fibonacci numbers")
    assert result is not None
    assert isinstance(result, dict)
    assert "best_solution" in result
    assert "exploration_path" in result 