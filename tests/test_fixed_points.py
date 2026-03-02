import torch

from rnntoolkit.fixed_points.fp import FixedPointCollection


def _make_fps():
    xstar = torch.tensor([[0.0, 0.1], [0.0, 0.1], [1.0, 1.0]])
    x_init = xstar + 0.1
    inputs = torch.tensor([[0.2], [0.2], [0.3]])
    qstar = torch.tensor([0.5, 0.2, 0.1])
    n_iters = torch.tensor([1, 1, 1])
    return FixedPointCollection(
        xstar=xstar,
        x_init=x_init,
        inputs=inputs,
        F_xstar=xstar,
        qstar=qstar,
        dq=qstar * 0.1,
        n_iters=n_iters,
        tol_unique=1e-3,
    )


def test_len_and_getitem_slice():
    fps = _make_fps()
    assert len(fps) == 3

    sub = fps[:2]
    assert sub.n == 2
    assert torch.equal(sub.xstar, fps.xstar[:2])


def test_setitem_updates_subset():
    fps = _make_fps()
    replacement = fps[1]
    replacement.xstar = torch.tensor([[9.0, 9.0]], dtype=torch.float32)

    fps[1] = replacement
    print(fps.xstar)
    assert torch.equal(fps.xstar[1], torch.tensor([9.0, 9.0]))


def test_contains_and_find_match():
    fps = _make_fps()
    fp_single = FixedPointCollection(
        xstar=fps.xstar[:1],
        x_init=fps.x_init[:1],
        inputs=fps.inputs[:1],
        F_xstar=fps.F_xstar[:1],
        qstar=fps.qstar[:1],
        dq=fps.dq[:1],
        n_iters=fps.n_iters[:1],
        tol_unique=fps.tol_unique,
    )

    assert fp_single in fps
    idx = fps.find(fp_single)
    assert idx.numel() == 2


def test_get_unique_prefers_lowest_qstar():
    fps = _make_fps()
    unique = fps.get_unique()

    assert unique.n == 2
    assert torch.allclose(unique.xstar[0], torch.tensor([0.0, 0.1]))
    assert torch.allclose(unique.qstar[0], torch.tensor(0.2))


def test_update_and_concatenate():
    fps = _make_fps()
    other = fps[:1]

    fps.update(other)
    assert fps.n == 4

    cat = FixedPointCollection.concatenate([fps[:1], fps[1:3]])
    assert cat.n == 3


def test_transform_and_restore(tmp_path):
    fps = _make_fps()
    U = torch.eye(2)
    offset = torch.tensor([1.0, -1.0])
    transformed = fps.transform(U, offset=offset)
    assert torch.allclose(transformed.xstar, fps.xstar + offset)

    path = tmp_path / "fps.pkl"
    fps.save(str(path))

    restored = _make_fps()
    restored.restore(str(path))
    assert torch.allclose(restored.xstar, fps.xstar)
