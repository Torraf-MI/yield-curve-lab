"""yield-curve-lab: curve dei tassi BTP e Bund, valore relativo dei titoli e rischio di un portafoglio
obbligazionario (key rate duration, scenari, VaR storico, copertura e immunizzazione)."""

from . import dati
from .curve import CurvaSpostata, Svensson
from .obbligazioni import Obbligazione, duration_effettiva, key_rate_duration
from .portafoglio import Portafoglio, Posizione
from .stima import CurvaStimata, stima_curva, stima_robusta

__all__ = ["dati", "CurvaSpostata", "CurvaStimata", "Obbligazione", "Portafoglio", "Posizione", "Svensson",
           "duration_effettiva", "key_rate_duration", "stima_curva", "stima_robusta"]
__version__ = "0.1.0"
