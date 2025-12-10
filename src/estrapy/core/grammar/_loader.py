from importlib.resources import files, as_file

def load_grammar(filename: str) -> str:
    """Loads a grammar file from package resources.
    If a single file is being run, loads from the filesystem instead."""
    
    if not __package__:
        from pathlib import Path
        return (Path(__file__).parent / filename).read_text()
    
    with as_file(files("estrapy.core.grammar") / filename) as grammar_path:
        return grammar_path.read_text()
