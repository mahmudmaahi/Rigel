from pydantic import BaseModel, Field

class AudioDenoiseRequest(BaseModel):
    method: str = Field(
        ..., 
        description="Denoising method: 'spectral_subtraction', 'wiener', 'logmmse', 'imcra'"
    )
    
    # Spectral Subtraction parameters
    alpha: float = Field(1.0, description="Oversubtraction factor (alpha)")
    beta: float = Field(0.01, description="Spectral floor (beta)")
    
    # Decision-Directed parameters (Wiener, Log-MMSE, OM-LSA)
    alpha_dd: float = Field(0.98, description="Decision-Directed a-priori SNR smoothing factor (alpha_dd)")
    
    # Minimum Statistics Noise Tracking (Phase 2) parameters
    noise_alpha_s: float = Field(0.98, description="Noise tracker temporal smoothing (alpha_s)")
    noise_bias: float = Field(1.5, description="Noise tracker empirical bias correction (B)")
    
    # IMCRA + OM-LSA specific parameters
    g_min: float = Field(0.05, description="OM-LSA minimum gain floor (G_min)")
    imcra_alpha_s: float = Field(0.86, description="IMCRA time smoothing (alpha_s)")
    imcra_alpha_d: float = Field(0.85, description="IMCRA base noise tracking rate (alpha_d)")
