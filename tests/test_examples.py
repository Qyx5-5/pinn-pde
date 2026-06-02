from examples.burgers_1d import make_trainer as make_burgers_trainer
from examples.schrodinger_1d import make_trainer as make_schrodinger_trainer


def test_burgers_example_smoke():
    trainer = make_burgers_trainer()
    record = trainer.step()
    assert record["loss"] >= 0.0


def test_schrodinger_example_smoke():
    trainer = make_schrodinger_trainer()
    record = trainer.step()
    assert record["loss"] >= 0.0
