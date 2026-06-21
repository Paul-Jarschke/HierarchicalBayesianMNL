import liesel.goose as gs

def run_nuts_inference_mixture_hbmnl(model, data_dict: dict, chains: int = 1, warmup: int = 1000, posterior: int = 5000, seed: int = 123):
    """
    Sets up and runs the Liesel/Goose MCMC engine for the Mixture HMNL model.
    Configures a block-wise NUTS strategy adapted to the mixture components.
    """
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(gs.LieselInterface(model))
    eb.set_initial_values(model.state)

    has_Z = data_dict.get("Z") is not None

    # Sample the latent variables 
    # pvec_latent controls component weights; sigma_inv_chol_k_latent controls component covariances
    # Use mm_diag=True to speed up NUTS sampling and only use diagonal elements instead of the full mass matrix
    eb.add_kernel(gs.NUTSKernel(["pvec_latent", "sigma_inv_chol_k_latent"], mm_diag=True))
    
    # Sample the component means
    eb.add_kernel(gs.NUTSKernel(["mu_k"])) 

    # Sample global demographic coefficients (if available)
    if has_Z:
        eb.add_kernel(gs.NUTSKernel(["Delta"]))

    # Sample the unit-level coefficients
    eb.add_kernel(gs.NUTSKernel(["beta_i"]))

    # Define warmup and posterior length
    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print("Starting NUTS Sampling for Mixture HMNL...")
    print(f" - Demographic covariates (Delta) included: {has_Z}")
    print(f" - Mixture components: {data_dict.get('K', 'Unknown')}")
    print(f" - Chains: {chains} | Warmup: {warmup} | Posterior: {posterior}")
    
    engine = eb.build()
    engine.sample_all_epochs()

    return engine.get_results(), engine.get_results().get_posterior_samples()