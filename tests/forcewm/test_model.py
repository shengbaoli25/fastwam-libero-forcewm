import copy
import pytest
import torch
from fastwam.forcewm.model import AutoregressiveForceWM, shifted_wrench_inputs, masked_force_mse
from fastwam.forcewm.masks import build_force_attention_mask
from fastwam.forcewm.integration import attach_forcewm
from helpers import model, sample

torch.set_num_threads(1)

def force(depth=3, checkpoint=False):
    torch.manual_seed(1)
    return AutoregressiveForceWM(video_hidden_dim=16,text_dim=12,hidden_dim=16,
        ffn_dim=32,num_heads=2,num_layers=depth,horizon=5,history_steps=4,
        checkpoint_layers=checkpoint)

def conditions(depth=3,batch=2):
    torch.manual_seed(2)
    return dict(history=torch.randn(batch,4,13),noisy_action=torch.randn(batch,5,7),
        action_clock=torch.ones(batch)*500,context=torch.randn(batch,3,12),
        context_mask=torch.ones(batch,3,dtype=torch.bool),history_is_pad=torch.zeros(batch,4,dtype=torch.bool),
        candidate_is_pad=torch.zeros(batch,5,dtype=torch.bool),history_channel_valid=torch.ones(batch,4,13,dtype=torch.bool),
        observation_latents=torch.randn(batch,48,1,2,2),video_features=tuple(torch.randn(batch,6,16) for _ in range(depth)),
        video_is_pad=torch.zeros(batch,6,dtype=torch.bool))

def enhanced(depth=3,checkpoint=False):
    return attach_forcewm(model(depth),allow_tiny_test_model=True,generated_video_steps=1,
        force_hidden_dim=16,force_ffn_dim=32,force_num_heads=2,memory_attention_width=8,
        memory_heads=2,checkpoint_force_layers=checkpoint)

def test_mask_topology_and_padding():
    h=torch.tensor([[True,False,False]])
    u=torch.tensor([[False,False,True]])
    w=torch.zeros((1,3),dtype=torch.bool)
    m=build_force_attention_mask(h,u,w)[0,0]
    assert not m[:3,3:].any() and not m[3:6,:3].any() and not m[3:6,6:].any()
    assert torch.equal(m[6:,6:],torch.ones(3,3,dtype=torch.bool).tril())
    assert not m[1:,0].any() and not m[[1,2,3,4,6,7,8],5].any()
    assert m.any(-1).all()

@pytest.mark.parametrize('depth',[1,3,30])
def test_future_teacher_perturbation_cannot_affect_earlier_predictions(depth):
    m=force(depth); c=conditions(depth)
    w=torch.randn(2,5,6)
    a=m.forward_teacher(previous_wrench=w,**c)
    changed=w.clone();changed[:,3:]+=999
    b=m.forward_teacher(previous_wrench=changed,**c)
    torch.testing.assert_close(a.wrench[:,:3],b.wrench[:,:3],rtol=0,atol=0)
    for x,y in zip(a.layers,b.layers):
        torch.testing.assert_close(x[:,:3],y[:,:3],rtol=0,atol=0)

@pytest.mark.parametrize('depth',[1,3,30])
def test_kv_cached_rollout_matches_full_prefix_operator(depth):
    m=force(depth).eval();c=conditions(depth)
    c['history_is_pad'][:,0]=True;c['history_channel_valid'][:,0]=False
    state=torch.get_rng_state().clone()
    result=m.rollout(**c)
    assert torch.equal(state,torch.get_rng_state())
    previous=torch.zeros(2,5,6);previous[:,0]=c['history'][:,-1,7:]
    with torch.no_grad():
        for i in range(5):
            full=m.forward_teacher(previous_wrench=previous,**c)
            torch.testing.assert_close(full.wrench[:,i],result.wrench[:,i],rtol=2e-5,atol=2e-6)
            for a,b in zip(full.layers,result.layers):
                torch.testing.assert_close(a[:,i],b[:,i],rtol=2e-5,atol=2e-6)
            if i+1<5:previous[:,i+1]=full.wrench[:,i]

def test_shift_and_strict_plus_one():
    c=conditions();target=torch.arange(60).reshape(2,5,6).float()
    p=shifted_wrench_inputs(c['history'],target,torch.zeros(2,5,dtype=torch.bool),torch.ones_like(target,dtype=torch.bool))
    assert torch.equal(p[:,0],c['history'][:,-1,7:])
    assert torch.equal(p[:,1:],target[:,:-1])
    target[:,-1]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        shifted_wrench_inputs(c['history'],target,torch.zeros(2,5,dtype=torch.bool),torch.ones_like(target,dtype=torch.bool))

@pytest.mark.parametrize('field',['wrench_future','target','noise','future_is_pad','previous_wrench'])
def test_rollout_rejects_future_and_noise(field):
    with pytest.raises(ValueError,match='does not accept'):
        force().rollout(**conditions(),**{field:torch.zeros(2,5,6)})

def test_checkpoint_backward_matches_eager_all_thirty_blocks():
    a=force(30);b=copy.deepcopy(a);b.checkpoint_layers=True
    c=conditions(30);w=torch.randn(2,5,6)
    x=a.forward_teacher(previous_wrench=w,**c).wrench.square().mean();x.backward()
    y=b.forward_teacher(previous_wrench=w,**c).wrench.square().mean();y.backward()
    torch.testing.assert_close(x,y)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert p.grad is not None and q.grad is not None,name
        torch.testing.assert_close(p.grad,q.grad)

def test_initialization_and_disabled_baseline_parity():
    base=model();ext=copy.deepcopy(base);state=torch.get_rng_state().clone()
    attach_forcewm(ext,allow_tiny_test_model=True,generated_video_steps=1,force_hidden_dim=16,
        force_ffn_dim=32,force_num_heads=2,memory_attention_width=8,memory_heads=2)
    assert torch.equal(state,torch.get_rng_state())
    for k,v in base.mot.state_dict().items():assert torch.equal(v,ext.mot.state_dict()[k])
    ext.force_enabled=False;s=sample()
    torch.manual_seed(5);loss,logs=base.training_loss(s)
    torch.manual_seed(5);other,otherlogs=ext.training_loss(s)
    assert torch.equal(loss,other) and logs==otherlogs

@pytest.mark.parametrize('batch',[1,2,16])
def test_real_operators_joint_loss_all_parameters_and_no_vae(batch):
    m=enhanced(checkpoint=True);s=sample(batch)
    def fail(*args,**kw):raise AssertionError('VAE called during cached training')
    m.vae.encode=fail;m.vae.decode=fail
    loss,logs=m.training_loss(s);loss.backward()
    assert torch.isfinite(loss) and set(logs)=={'loss_video','loss_action','loss_force'}
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.mot.forcewm.parameters())
    assert all(p.grad is not None for p in m.mot.action_dual_memory.parameters())

def test_teacher_force_never_leaks_to_action_even_with_open_gates():
    a=enhanced();b=copy.deepcopy(a)
    for m in (a,b):
        for layer in m.mot.action_dual_memory:layer.force_gate.data.fill_(.5)
    s=sample();t=copy.deepcopy(s);t['wrench_future']+=50
    torch.manual_seed(12);_,x=a.training_loss(s)
    torch.manual_seed(12);_,y=b.training_loss(t)
    assert x['loss_action']==y['loss_action'] and x['loss_video']==y['loss_video']
    assert x['loss_force']!=y['loss_force']

def test_inference_does_not_read_future_labels():
    a=enhanced();b=copy.deepcopy(a)
    s=sample(1);t=copy.deepcopy(s)
    for key in ('action','wrench_future','wrench_valid','force_is_pad'):
        t[key]=torch.full_like(t[key],float('nan')) if t[key].is_floating_point() else ~t[key]
    t['video_latents'][:,:,1:]=float('nan')
    x=a.infer_forcewm(s,num_inference_steps=2,seed=4)
    y=b.infer_forcewm(t,num_inference_steps=2,seed=4)
    for key in ('action','wrench','video_latents'):assert torch.equal(x[key],y[key])

def test_strict_checkpoint_and_advanced_rng_roundtrip(tmp_path):
    a=enhanced();a.memory_generator.normal((3,))
    path=tmp_path/'weights.pt';a.save_checkpoint(path,step=99)
    b=enhanced();result=b.load_checkpoint(path)
    assert result['step']==99
    assert torch.equal(a.memory_generator.get_state(),b.memory_generator.get_state())
    for k,v in a.mot.state_dict().items():assert torch.equal(v,b.mot.state_dict()[k])
    b.forcewm_manifest['lambda_force']=9
    with pytest.raises(ValueError,match='manifest differs'):b.load_checkpoint(path)
