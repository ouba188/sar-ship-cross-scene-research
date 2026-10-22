"""Matched retraining, not promotion of a post-hoc path deletion."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import train_signed

if __name__=='__main__':
    train_signed.CACHE=HERE.parent
    train_signed.HERE=HERE
    train_signed.ARMS=['scalar_signed','signed_only']
    train_signed.main()
