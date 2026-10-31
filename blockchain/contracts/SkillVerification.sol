// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title SkillVerification
/// @notice Anchors SHA-256 hashes of candidate skill profiles on-chain for
///         tamper-evident verification (Section 3.12 of the accompanying paper).
///         Only the hash and an off-chain metadata pointer (e.g. an IPFS/URI
///         reference) are stored on-chain; the raw resume text and the extracted
///         skill list never touch the chain.
/// @dev    Access model: a single registrar (the deployer) may anchor profiles;
///         the registrar grants/revokes read/verification access to employer
///         addresses. A stored record cannot be overwritten (immutability).
contract SkillVerification {
    struct ProfileRecord {
        bytes32 profileHash;
        string metadataURI;
        uint256 timestamp;
        bool exists;
    }

    address public immutable owner;
    mapping(address => bool) private authorized;
    mapping(string => ProfileRecord) private records;

    event ProfileStored(string indexed candidateId, bytes32 profileHash, string metadataURI, uint256 timestamp);
    event AccessGranted(address indexed account);
    event AccessRevoked(address indexed account);

    modifier onlyOwner() {
        require(msg.sender == owner, "not registrar");
        _;
    }

    modifier onlyAuthorized() {
        require(msg.sender == owner || authorized[msg.sender], "not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function GrantAccess(address account) external onlyOwner {
        authorized[account] = true;
        emit AccessGranted(account);
    }

    function RevokeAccess(address account) external onlyOwner {
        authorized[account] = false;
        emit AccessRevoked(account);
    }

    /// @notice Writes a new immutable record keyed by candidate ID.
    function StoreProfile(string calldata candidateId, bytes32 profileHash, string calldata metadataURI) external onlyOwner {
        require(!records[candidateId].exists, "profile already stored");
        records[candidateId] = ProfileRecord({
            profileHash: profileHash,
            metadataURI: metadataURI,
            timestamp: block.timestamp,
            exists: true
        });
        emit ProfileStored(candidateId, profileHash, metadataURI, block.timestamp);
    }

    /// @notice Compares a submitted hash against the stored record.
    function VerifyProfile(string calldata candidateId, bytes32 profileHash) external view onlyAuthorized returns (bool) {
        ProfileRecord storage r = records[candidateId];
        require(r.exists, "profile not found");
        return r.profileHash == profileHash;
    }

    /// @notice Returns the stored hash and metadata pointer for an authorized caller.
    function FetchProfile(string calldata candidateId) external view onlyAuthorized returns (bytes32 profileHash, string memory metadataURI, uint256 timestamp) {
        ProfileRecord storage r = records[candidateId];
        require(r.exists, "profile not found");
        return (r.profileHash, r.metadataURI, r.timestamp);
    }
}
