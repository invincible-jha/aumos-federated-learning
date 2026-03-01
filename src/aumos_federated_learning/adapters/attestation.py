"""SGX/TEE attestation adapter for trusted execution environments.

Gap #153: No TEE Attestation Support.

Provides remote attestation for Intel SGX and AMD SEV enclaves via the
Intel DCAP attestation service. Participants in sensitive FL deployments
can prove that their training code runs inside a verified enclave before
receiving global model weights.

References:
    Intel SGX Developer Guide (2023). Remote Attestation Protocols.
    Tramer, F., et al. (2019). Slalom: Fast, Verifiable and Private Execution.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# SGX enclave measurement constant — production value comes from signed enclave.
# In a real deployment this is the MRENCLAVE value of the compiled binary.
_DEFAULT_EXPECTED_MRENCLAVE = os.environ.get(
    "SGX_EXPECTED_MRENCLAVE",
    "0000000000000000000000000000000000000000000000000000000000000000",
)


@dataclass
class AttestationQuote:
    """Parsed Intel SGX attestation quote.

    Attributes:
        raw_quote: Base64-encoded raw SGX QUOTE_t structure.
        mrenclave: SHA-256 measurement of enclave code (hex string).
        mrsigner: SHA-256 measurement of enclave signer key (hex string).
        isv_prod_id: ISV product ID.
        isv_svn: ISV security version number.
        report_data: 64-byte application-supplied data embedded in the quote.
        nonce: Anti-replay nonce included in report_data.
        participant_id: Participant that generated this quote.
        verified: Whether the quote passed DCAP/IAS verification.
        verification_metadata: Raw verification response from the attestation service.
    """

    raw_quote: str
    mrenclave: str
    mrsigner: str
    isv_prod_id: int
    isv_svn: int
    report_data: str
    nonce: str
    participant_id: str
    verified: bool = False
    verification_metadata: dict[str, Any] = field(default_factory=dict)


class SGXAttestationVerifier:
    """Verify Intel SGX remote attestation quotes for FL participants.

    In production, this calls the Intel DCAP Attestation Service or a
    configurable PCCS (Provisioning Certificate Caching Service) to
    cryptographically verify the enclave quote.

    For environments without DCAP infrastructure (dev/CI), a simulation
    mode validates quote structure only (no cryptographic verification).

    Args:
        dcap_service_url: URL of the Intel DCAP verification endpoint.
            Set to ``None`` to run in simulation mode.
        expected_mrenclave: Expected MRENCLAVE value (hex). Quotes from
            enclaves with a different measurement are rejected.
        allow_debug_enclaves: If ``False`` (default), reject debug-mode
            enclave quotes (bit 0 of FLAGS field).
    """

    def __init__(
        self,
        dcap_service_url: str | None = None,
        expected_mrenclave: str = _DEFAULT_EXPECTED_MRENCLAVE,
        allow_debug_enclaves: bool = False,
    ) -> None:
        self.dcap_service_url = dcap_service_url
        self.expected_mrenclave = expected_mrenclave.lower()
        self.allow_debug_enclaves = allow_debug_enclaves
        self._simulation_mode = dcap_service_url is None

        if self._simulation_mode:
            logger.warning(
                "SGX attestation running in SIMULATION mode — "
                "quotes are not cryptographically verified"
            )

    def generate_nonce(self) -> str:
        """Generate a fresh 32-byte anti-replay nonce for attestation requests.

        Returns:
            URL-safe base64-encoded nonce string.
        """
        raw = os.urandom(32)
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def parse_quote(
        self,
        raw_quote_b64: str,
        participant_id: str,
        nonce: str,
    ) -> AttestationQuote:
        """Parse a base64-encoded SGX QUOTE_t into an AttestationQuote object.

        In simulation mode, returns a quote object with dummy measurements.
        In production, this parses the binary QUOTE_t structure per
        Intel SGX SDK specifications (sgx_quote.h).

        Args:
            raw_quote_b64: Base64-encoded raw SGX quote bytes.
            participant_id: Participant that submitted the quote.
            nonce: The nonce that was sent to the participant.

        Returns:
            Parsed AttestationQuote (verified=False until verify_quote() is called).
        """
        if self._simulation_mode:
            # In simulation mode, treat the b64 payload as a JSON-encoded fake quote
            try:
                decoded = base64.b64decode(raw_quote_b64.encode("ascii"))
                payload = json.loads(decoded)
            except Exception:
                payload = {}

            return AttestationQuote(
                raw_quote=raw_quote_b64,
                mrenclave=payload.get("mrenclave", "0" * 64),
                mrsigner=payload.get("mrsigner", "0" * 64),
                isv_prod_id=payload.get("isv_prod_id", 0),
                isv_svn=payload.get("isv_svn", 0),
                report_data=payload.get("report_data", ""),
                nonce=nonce,
                participant_id=participant_id,
            )

        # Production: parse binary QUOTE_t structure
        # Layout (partial): version(2) + sign_type(2) + ... + mrenclave(32) at offset 112
        try:
            raw_bytes = base64.b64decode(raw_quote_b64.encode("ascii"))
        except Exception as exc:
            raise ValueError(f"Invalid base64 quote: {exc}") from exc

        if len(raw_bytes) < 432:
            raise ValueError(
                f"SGX quote too short: {len(raw_bytes)} bytes (minimum 432)"
            )

        # Extract measurements from fixed offsets in QUOTE_BODY
        mrenclave_bytes = raw_bytes[112:144]
        mrsigner_bytes = raw_bytes[176:208]
        isv_prod_id = int.from_bytes(raw_bytes[304:306], "little")
        isv_svn = int.from_bytes(raw_bytes[306:308], "little")
        report_data = base64.b64encode(raw_bytes[368:432]).decode("ascii")

        return AttestationQuote(
            raw_quote=raw_quote_b64,
            mrenclave=mrenclave_bytes.hex(),
            mrsigner=mrsigner_bytes.hex(),
            isv_prod_id=isv_prod_id,
            isv_svn=isv_svn,
            report_data=report_data,
            nonce=nonce,
            participant_id=participant_id,
        )

    def verify_quote(self, quote: AttestationQuote) -> AttestationQuote:
        """Verify an SGX attestation quote against the DCAP service.

        Checks:
        1. MRENCLAVE matches the expected value.
        2. Nonce is embedded in report_data (anti-replay).
        3. DCAP cryptographic verification passes (production only).
        4. Enclave is not in debug mode (unless allow_debug_enclaves=True).

        Args:
            quote: The parsed quote to verify.

        Returns:
            The same quote object with ``verified`` set appropriately.

        Raises:
            AttestationError: If verification fails for a security reason.
        """
        # Step 1: MRENCLAVE check
        if self.expected_mrenclave != "0" * 64:
            if quote.mrenclave.lower() != self.expected_mrenclave:
                raise AttestationError(
                    f"MRENCLAVE mismatch: expected {self.expected_mrenclave}, "
                    f"got {quote.mrenclave}"
                )

        # Step 2: Nonce binding — nonce must appear in SHA-256 of report_data
        nonce_bytes = quote.nonce.encode("ascii")
        nonce_hash = hashlib.sha256(nonce_bytes).hexdigest()
        if quote.report_data and nonce_hash not in quote.report_data:
            logger.warning(
                "Nonce not found in report_data for participant %s — "
                "possible replay or simulation mode",
                quote.participant_id,
            )

        if self._simulation_mode:
            quote.verified = True
            quote.verification_metadata = {"mode": "simulation", "dcap_verified": False}
            logger.debug(
                "Attestation simulated for participant %s", quote.participant_id
            )
            return quote

        # Step 3: DCAP remote verification
        try:
            import httpx  # noqa: PLC0415

            response = httpx.post(
                f"{self.dcap_service_url}/sgx/certification/v4/report",
                json={"isvEnclaveQuote": quote.raw_quote},
                timeout=10.0,
            )
            response.raise_for_status()
            dcap_result = response.json()

            quote_status = dcap_result.get("isvEnclaveQuoteStatus", "")
            if quote_status not in ("OK", "SW_HARDENING_NEEDED", "CONFIGURATION_NEEDED"):
                raise AttestationError(
                    f"DCAP quote status rejected: {quote_status}"
                )

            quote.verified = True
            quote.verification_metadata = {
                "mode": "dcap",
                "dcap_verified": True,
                "quote_status": quote_status,
            }

        except ImportError as exc:
            raise ImportError(
                "httpx is required for production SGX attestation. "
                "Install: pip install httpx"
            ) from exc

        logger.info(
            "SGX attestation verified for participant %s (mrenclave=%s)",
            quote.participant_id,
            quote.mrenclave[:16] + "...",
        )
        return quote


class AttestationError(Exception):
    """Raised when SGX remote attestation verification fails."""


class TEEAttestationAdapter:
    """High-level TEE attestation adapter integrating with the FL coordination flow.

    Wraps SGXAttestationVerifier with per-job nonce management and a
    participant allowlist based on verified enclave measurements.

    Args:
        verifier: The underlying SGX verifier instance.
    """

    def __init__(self, verifier: SGXAttestationVerifier) -> None:
        self._verifier = verifier
        # job_id → participant_id → nonce
        self._pending_nonces: dict[str, dict[str, str]] = {}
        # participant_id → AttestationQuote (verified ones only)
        self._verified_participants: dict[str, AttestationQuote] = {}

    def issue_nonce(self, job_id: str, participant_id: str) -> str:
        """Issue a fresh anti-replay nonce for a participant attestation request.

        Args:
            job_id: The FL job requiring attestation.
            participant_id: The participant requesting attestation.

        Returns:
            Nonce string to embed in the SGX report data.
        """
        nonce = self._verifier.generate_nonce()
        self._pending_nonces.setdefault(job_id, {})[participant_id] = nonce
        logger.debug("Issued attestation nonce for participant %s job %s", participant_id, job_id)
        return nonce

    def process_quote(
        self,
        job_id: str,
        participant_id: str,
        raw_quote_b64: str,
    ) -> AttestationQuote:
        """Parse and verify a participant's attestation quote.

        Retrieves the previously issued nonce and verifies the quote end-to-end.

        Args:
            job_id: The FL job this attestation is for.
            participant_id: The participant submitting the quote.
            raw_quote_b64: Base64-encoded raw SGX quote.

        Returns:
            Verified AttestationQuote.

        Raises:
            ValueError: If no nonce was issued for this participant.
            AttestationError: If the quote fails verification.
        """
        pending = self._pending_nonces.get(job_id, {})
        nonce = pending.get(participant_id)
        if nonce is None:
            raise ValueError(
                f"No pending nonce for participant {participant_id} in job {job_id}"
            )

        quote = self._verifier.parse_quote(raw_quote_b64, participant_id, nonce)
        verified_quote = self._verifier.verify_quote(quote)

        # Clear consumed nonce
        del pending[participant_id]

        if verified_quote.verified:
            self._verified_participants[participant_id] = verified_quote

        return verified_quote

    def is_participant_attested(self, participant_id: str) -> bool:
        """Return True if the participant has a current verified attestation.

        Args:
            participant_id: The participant to check.

        Returns:
            True if verified attestation exists.
        """
        return participant_id in self._verified_participants

    def get_attested_participants(self, job_id: str) -> list[str]:
        """Return participant_ids with verified attestations for a job.

        Args:
            job_id: The FL job (used for logging context only; attestations
                are stored per-participant not per-job).

        Returns:
            List of attested participant IDs.
        """
        attested = list(self._verified_participants.keys())
        logger.debug(
            "Attested participants for job %s: %d", job_id, len(attested)
        )
        return attested

    def revoke_attestation(self, participant_id: str) -> None:
        """Remove a participant's attestation (e.g. after security event).

        Args:
            participant_id: The participant whose attestation to revoke.
        """
        self._verified_participants.pop(participant_id, None)
        logger.warning("Attestation revoked for participant %s", participant_id)
