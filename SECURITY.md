# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Please do NOT report security vulnerabilities via public GitHub issues.

Email: security@muveraai.com

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You will receive a response within 48 hours. We follow responsible disclosure with a 90-day remediation window.

## Security Considerations

### Federated Learning Attack Surface

- **Model poisoning**: Participants may submit malicious model updates. The secure aggregation and differential privacy layers provide defenses.
- **Inference attacks**: Aggregated model updates may leak information about participant training data. Differential privacy with formal (ε, δ) guarantees mitigates this.
- **Byzantine participants**: FedProx and SCAFFOLD strategies include robustness to non-IID and potentially adversarial participants.

### Cryptographic Operations

- Secure aggregation uses MPC-based protocols. Private keys are never transmitted.
- All inter-service communication must use TLS in production.
- Flower gRPC connections should use mutual TLS (`FLOWER_SSL_ENABLED=true`) in production.

### Data Handling

- No raw training data ever leaves the participant's environment.
- Only model gradients/updates are transmitted, and these are protected by differential privacy noise.
- Aggregated model artifacts are stored in the configured object store; access is tenant-scoped via RLS.
