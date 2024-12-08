import names
import random
import hashlib
import asyncio
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, APIRouter
from typing import Optional, Dict, List, Tuple, Union, Any


def key_generation(n: int) -> Tuple[List[float], List[float], List[float]]:
    S = np.random.rand(n) + 1j * np.random.rand(n)
    M = np.random.rand(n)
    P = np.fft.fft(S) * M
    return S, P, M


def encrypt(plaintext: List[int], P: List[float]) -> np.ndarray:
    X = np.array(plaintext)
    X_f = np.fft.fft(X)
    C = X_f * P
    return C


def decrypt(ciphertext: np.ndarray, S: List[float], M: List[float]) -> np.ndarray:
    X_f = ciphertext / (np.fft.fft(S) * M)
    X = np.fft.ifft(X_f)
    return np.round(X).astype(int)


# Symmetric Encryption Functions
def fft_encrypt(data: np.ndarray, key: np.ndarray) -> np.ndarray:
    fft_data = np.fft.fft(data)
    fft_key = np.fft.fft(key)
    encrypted_fft = fft_data * fft_key
    return encrypted_fft


def fft_decrypt(encrypted_fft: np.ndarray, key: np.ndarray) -> np.ndarray:
    fft_key = np.fft.fft(key)
    decrypted_fft = encrypted_fft / fft_key
    decrypted_data = np.fft.ifft(decrypted_fft).real
    return decrypted_data


def serialize_complex_array(array: np.ndarray) -> List[Dict[str, float]]:
    """Convert a complex numpy array to a list of dictionaries with real and imaginary parts."""
    return [{"real": float(c.real), "imag": float(c.imag)} for c in array]


def deserialize_complex_array(array: List[Dict[str, float]]) -> np.ndarray:
    """Convert a list of dictionaries with real and imaginary parts back to a complex numpy array."""
    return np.array(
        [complex(item["real"], item["imag"]) for item in array], dtype=complex
    )


class Wallet(BaseModel):
    name: str
    private_key: Optional[np.ndarray] = Field(default=None)
    public_key: Optional[np.ndarray] = Field(default=None)
    mask: Optional[np.ndarray] = Field(default=None)

    def __init__(self, name: str, key_size: int = 8):
        super().__init__(name=name)
        self.private_key, self.public_key, self.mask = key_generation(key_size)

    def sign_transaction(self, transaction: List[int]):
        return encrypt(transaction, self.public_key)

    def decrypt_transaction(self, encrypted_transaction: np.ndarray):
        return decrypt(encrypted_transaction, self.private_key, self.mask)

    def model_dump(self, **kwargs):
        # Serialize model data to a JSON-friendly format
        return {
            "name": self.name,
            "public_key": self._safe_serialize(self.public_key),
        }

    @staticmethod
    def _safe_serialize(array):
        # Handle serialization of numpy arrays
        if isinstance(array, np.ndarray):
            # Ensure the array has no complex numbers
            if np.iscomplexobj(array):
                array = np.real(array)
            return array.tolist()
        return None

    class Config:
        arbitrary_types_allowed = True


class Transaction(BaseModel):
    sender: Wallet
    receiver: Wallet
    amount: float
    encrypted: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    def model_dump(self, **kwargs):
        # Serialize model data to a JSON-friendly format
        return {
            "sender": self.sender.name,
            "receiver": self.receiver.name,
            "amount": self.amount,
            "encrypted": serialize_complex_array(self.encrypted),
        }


class Block(BaseModel):
    transactions: List[Transaction]
    previous_hash: str
    nonce: int
    hash: str = ""

    def compute_hash(self):
        block_string = f"{self.transactions}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def model_dump(self, **kwargs):
        return {
            "transactions": [t.model_dump() for t in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
        }


class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty: int = 3  # Lower difficulty for faster mining

    def create_transaction(
        self, sender: Wallet, receiver: Wallet, amount: Union[int, float]
    ):
        transaction = Transaction(
            sender=sender,
            receiver=receiver,
            amount=amount,
            encrypted=sender.sign_transaction([amount] + [0] * 7),
        )
        self.pending_transactions.append(transaction)
        return transaction

    def verify_transaction(
        self, transaction: Transaction, sender_wallet: Wallet
    ) -> bool:
        decrypted = sender_wallet.decrypt_transaction(transaction.encrypted)
        return decrypted[0] == transaction.amount

    def mine_block(self, miner_wallet: Wallet) -> Block:
        # Prepare the block with pending transactions
        block = Block(
            transactions=self.pending_transactions.copy(),
            previous_hash=self.chain[-1].hash if self.chain else "0" * 64,
            nonce=0,
        )

        # Increment nonce until the hash satisfies the difficulty
        while True:
            block.hash = block.compute_hash()
            if block.hash.startswith("0" * self.difficulty):
                break
            block.nonce += 1

        # Add the mined block to the chain
        self.chain.append(block)
        self.pending_transactions.clear()

        # Reward the miner
        reward_transaction = Transaction(
            sender=Wallet("Network"),
            receiver=miner_wallet,
            amount=1,
            encrypted=miner_wallet.sign_transaction([1] + [0] * 7),
        )
        self.pending_transactions.append(reward_transaction)

        return block


app = FastAPI()
blockchain = Blockchain()
wallets = [Wallet("Alice"), Wallet("Bob")]

w_router = APIRouter(tags=["wallets"])
b_router = APIRouter(tags=["blockchain"])


@app.get("/create-dummies")
def create_dummies():
    # Create dummy wallets, and transactions for those wallets
    wallets.clear()
    [wallets.append(Wallet(names.get_first_name())) for _ in range(10)]
    for i in range(10):
        blockchain.create_transaction(
            wallets[i], wallets[(i + 1) % 10], random.randint(1, 1000)
        )
    return {"message": "Dummies created."}


@app.get("/set-difficulty")
def set_difficulty(difficulty: int = 3):
    blockchain.difficulty = difficulty
    return {"message": f"Difficulty set to {difficulty}."}


@w_router.get("/wallets")
async def get_wallets():
    return {"wallets": wallets}


@w_router.post("/create_wallet")
async def create_wallet(name: str):
    # Check if the wallet name already exists
    if name in [wallet.name for wallet in wallets]:
        return {"error": f"Wallet '{name}' already exists."}

    # Prevent using the reserved name "Network"
    if name == "Network":
        return {"error": "Invalid wallet name."}

    wallet = Wallet(name)
    wallets.append(wallet)
    return {"wallet": wallet}


@b_router.get("/blockchain")
async def get_blockchain():
    return {"blockchain": blockchain.chain}


@b_router.post("/create_transaction")
async def create_transaction(sender_name: str, receiver_name: str, amount: int):
    # Validate sender and receiver
    sender = next((wallet for wallet in wallets if wallet.name == sender_name), None)
    receiver = next(
        (wallet for wallet in wallets if wallet.name == receiver_name), None
    )

    if not sender:
        return {"error": f"Sender '{sender_name}' not found."}
    if not receiver:
        return {"error": f"Receiver '{receiver_name}' not found."}
    if amount <= 0:
        return {"error": "Amount must be greater than zero."}

    # Create transaction
    transaction = blockchain.create_transaction(sender, receiver, amount)
    return {"transaction": transaction}


@b_router.post("/verify_transaction")
async def verify_transaction(transaction: Dict[str, Any]):
    sender_name = transaction.get("sender")
    if not sender_name:
        return {"error": "Sender name is required in the transaction."}

    # Retrieve the sender's wallet
    sender_wallet = next(
        (wallet for wallet in wallets if wallet.name == sender_name), None
    )
    if not sender_wallet:
        return {"error": f"Sender '{sender_name}' not found."}

    # Retrieve the receiver's wallet
    receiver_name = transaction.get("receiver")
    if not receiver_name:
        return {"error": "Receiver name is required in the transaction."}

    receiver_wallet = next(
        (wallet for wallet in wallets if wallet.name == receiver_name), None
    )
    if not receiver_wallet:
        return {"error": f"Receiver '{receiver_name}' not found."}

    # Deserialize the encrypted data
    encrypted_data = transaction.get("encrypted")
    if encrypted_data is None:
        return {"error": "Encrypted transaction data is required."}

    try:
        encrypted_data = deserialize_complex_array(encrypted_data)
    except Exception as e:
        return {"error": f"Invalid encrypted data: {e}"}

    # Verify the transaction
    try:
        transaction = Transaction(
            sender=sender_wallet,
            receiver=receiver_wallet,
            amount=transaction["amount"],
            encrypted=encrypted_data,
        )
        is_valid = blockchain.verify_transaction(transaction, sender_wallet)
    except Exception as e:
        return {"error": f"Verification failed: {e}"}

    return {"valid": bool(is_valid)}


@b_router.post("/mine_block")
async def mine_block(miner: str):
    """
    Mines a new block and adds it to the blockchain asynchronously.

    Returns:
        dict: The mined block details.
    """
    try:
        # Retrieve the miner's wallet
        miner_wallet = next(
            (wallet for wallet in wallets if wallet.name == miner), None
        )
        if not miner_wallet:
            return {"error": f"Miner '{miner}' not found."}

        # Mine a block asynchronously
        block = await asyncio.to_thread(blockchain.mine_block, miner_wallet)

        print(block)

        return {"block": block}
    except Exception as e:
        return {"error": f"Failed to mine block: {e}"}


app.include_router(w_router)
app.include_router(b_router)
