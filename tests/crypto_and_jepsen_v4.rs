use authority_lab::{
    crypto_authority::{
        CanonicalEffect, CanonicalProposal, CapabilityMacaroon, CryptoSinkVerifier, Ed25519Keypair,
        MacaroonCaveat, MacaroonRoot, QuorumCertificate, Sha256, SignedFenceCert,
        ed25519_public_key, ed25519_sign, ed25519_verify, hex_decode, hex_encode, hmac_sha256,
        sha256, sha512,
    },
    distributed_sinks::crypto_and_jepsen_suite,
};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn fips_180_4_sha256_and_sha512_official_vectors() {
    // FIPS 180-4 SHA-256 vectors
    assert_eq!(
        hex_encode(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex_encode(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex_encode(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );

    // 1,000,000 'a's streaming chunked test
    let mut stream = Sha256::new();
    let chunk = [b'a'; 1000];
    for _ in 0..1000 {
        stream.update(&chunk);
    }
    assert_eq!(
        hex_encode(&stream.finalize()),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
    );

    // FIPS 180-4 SHA-512 vectors
    assert_eq!(
        hex_encode(&sha512(b"")),
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
         47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    );
    assert_eq!(
        hex_encode(&sha512(b"abc")),
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39\
         a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
    );
    assert_eq!(
        hex_encode(&sha512(
            b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
              ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
        )),
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018\
         501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
    );
}

#[test]
fn rfc_4231_hmac_sha256_official_vectors() {
    // Test Case 1
    let k1 = [0x0bu8; 20];
    assert_eq!(
        hex_encode(&hmac_sha256(&k1, b"Hi There")),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    );

    // Test Case 2
    assert_eq!(
        hex_encode(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    );

    // Test Case 3
    let k3 = [0xaau8; 20];
    let d3 = [0xddu8; 50];
    assert_eq!(
        hex_encode(&hmac_sha256(&k3, &d3)),
        "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"
    );

    // Test Case 6: key larger than 64-byte block size (131 bytes)
    let k6 = [0xaau8; 131];
    assert_eq!(
        hex_encode(&hmac_sha256(
            &k6,
            b"Test Using Larger Than Block-Size Key - Hash Key First"
        )),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
    );

    // Test Case 7: key larger than block size + multi-block message
    assert_eq!(
        hex_encode(&hmac_sha256(
            &k6,
            b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm."
        )),
        "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
    );
}

#[test]
fn rfc_8032_ed25519_official_vectors_and_openssl_cli_cross_check() {
    let sha_abc_msg = hex_decode(
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39\
         a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
    )
    .expect("hex");
    // RFC 8032 §7.1 TEST 1024 key derivation + 1024-byte multi-block message verification
    let seed_1024 =
        hex_decode("f5e5767cf153319517630f226876b86c8160cc583bc013744c6bf255f5cc0ee5")
            .expect("seed 1024");
    let mut seed_1024_arr = [0u8; 32];
    seed_1024_arr.copy_from_slice(&seed_1024);
    let pk_1024 = ed25519_public_key(&seed_1024_arr);
    assert_eq!(
        hex_encode(&pk_1024),
        "278117fc144c72340f67d0f2316e8386ceffbf2b2428c9c51fef7c597f1d426e"
    );
    let msg_1024_bytes: Vec<u8> = (0u16..1024).map(|i| (i.wrapping_mul(37) & 0xff) as u8).collect();
    let sig_1024 = ed25519_sign(&seed_1024_arr, &msg_1024_bytes);
    assert!(ed25519_verify(&pk_1024, &msg_1024_bytes, &sig_1024));

    let vectors: [(&str, &str, &[u8], &str); 4] = [
        (
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            b"",
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155\
             5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        ),
        (
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            &[0x72],
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da\
             085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
        ),
        (
            "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            &[0xaf, 0x82],
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac\
             18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
        ),
        (
            "833fe62409237b9d62ec77587520911e9a759cec1d19755b7da901b96dca3d42",
            "ec172b93ad5e563bf4932c70e1245034c35467ef2efd4d64ebf819683467e2bf",
            &sha_abc_msg,
            "dc2a4459e7369633a52b1bf277839a00201009a3efbf3ecb69bea2186c26b589\
             09351fc9ac90b3ecfdfbc7c66431e0303dca179c138ac17ad9bef1177331a704",
        ),
    ];

    for (seed_hex, pk_hex, msg, sig_hex) in vectors {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&hex_decode(seed_hex).expect("seed"));
        let pk = ed25519_public_key(&seed);
        assert_eq!(hex_encode(&pk), pk_hex);

        let sig = ed25519_sign(&seed, msg);
        assert_eq!(hex_encode(&sig), sig_hex);
        assert!(ed25519_verify(&pk, msg, &sig));

        // Reject non-canonical S >= L (set top nibble of S to 0xff)
        let mut non_canonical = sig;
        non_canonical[63] = 0xff;
        assert!(!ed25519_verify(&pk, msg, &non_canonical));

        // Reject exact boundary S == L (2^252 + 27742317777372353535851937790883648493)
        let l_le = hex_decode("edd3f55c1a631258d69cf7a2def9de1400000000000000000000000000000010")
            .expect("L");
        let mut s_eq_l = sig;
        s_eq_l[32..64].copy_from_slice(&l_le);
        assert!(!ed25519_verify(&pk, msg, &s_eq_l));

        // Reject non-canonical field element R.y == p (2^255 - 19)
        let p_le = hex_decode("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f")
            .expect("p");
        let mut r_eq_p = sig;
        r_eq_p[0..32].copy_from_slice(&p_le);
        assert!(!ed25519_verify(&pk, msg, &r_eq_p));

        // Reject tampered message
        let mut alt_msg = msg.to_vec();
        alt_msg.push(0x01);
        assert!(!ed25519_verify(&pk, &alt_msg, &sig));
    }

    // Wycheproof small-order 8-torsion public key & R-nonce forgery vectors:
    // Every 8-torsion point on Ed25519 must be rejected as public key or R nonce even when S = 0.
    let torsion_points_hex = [
        "0100000000000000000000000000000000000000000000000000000000000000", // (0, 1) order 1
        "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f", // (0, -1) order 2
        "0000000000000000000000000000000000000000000000000000000000000000", // (sqrt(-1), 0) order 4
        "0000000000000000000000000000000000000000000000000000000000000080", // (-sqrt(-1), 0) order 4
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05", // order 8
        "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc85", // order 8
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a", // order 8
        "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac03fa", // order 8
    ];
    for tp_hex in torsion_points_hex {
        let mut tp = [0u8; 32];
        tp.copy_from_slice(&hex_decode(tp_hex).expect("torsion hex"));
        let mut zero_sig = [0u8; 64];
        zero_sig[..32].copy_from_slice(&tp); // R = torsion, S = 0
        assert!(
            !ed25519_verify(&tp, b"wycheproof-small-order-forgery", &zero_sig),
            "must reject small-order torsion point {tp_hex}"
        );
    }

    // Live bidirectional cross-verification with OpenSSL 3+ CLI (mandatory on Linux; skipped if macOS LibreSSL)
    let ossl_version = Command::new("/usr/bin/openssl")
        .arg("version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    if cfg!(target_os = "linux") {
        assert!(
            ossl_version.starts_with("OpenSSL 3."),
            "expected OpenSSL 3.x on Linux, got {ossl_version:?}"
        );
    }
    if !ossl_version.starts_with("OpenSSL 3.") {
        return;
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let scratch = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("v4-systems")
        .join(format!("openssl-check-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&scratch).expect("mkdir scratch");

    let pkcs8_prefix = hex_decode("302e020100300506032b657004220420").expect("pkcs8");
    let spki_prefix = hex_decode("302a300506032b6570032100").expect("spki");

    // 1. Generate keypair in pure Rust, sign payload, verify with `/usr/bin/openssl pkeyutl -verify`
    let kp = Ed25519Keypair::deterministic("openssl.interop.test", 7);
    let msg = b"jeath.quorum.v4 bidirectional OpenSSL 3.0 Ed25519 verification payload";
    let rust_sig = kp.sign(msg);

    let pk_der_path = scratch.join("rust_pub.der");
    let sk_der_path = scratch.join("rust_priv.der");
    let msg_path = scratch.join("msg.bin");
    let rust_sig_path = scratch.join("rust_sig.bin");
    let ossl_sig_path = scratch.join("ossl_sig.bin");

    let mut spki = spki_prefix.clone();
    spki.extend_from_slice(&kp.public_key);
    fs::write(&pk_der_path, &spki).expect("write spki");

    let mut pkcs8 = pkcs8_prefix;
    pkcs8.extend_from_slice(&kp.seed);
    fs::write(&sk_der_path, &pkcs8).expect("write pkcs8");
    fs::write(&msg_path, msg).expect("write msg");
    fs::write(&rust_sig_path, rust_sig.0).expect("write rust sig");

    let verify_status = Command::new("/usr/bin/openssl")
        .args(["pkeyutl", "-verify", "-pubin", "-inkey"])
        .arg(&pk_der_path)
        .args(["-keyform", "DER", "-rawin", "-in"])
        .arg(&msg_path)
        .arg("-sigfile")
        .arg(&rust_sig_path)
        .status()
        .expect("invoke /usr/bin/openssl pkeyutl -verify");
    assert!(
        verify_status.success(),
        "OpenSSL failed to verify pure-Rust Ed25519 signature"
    );

    // 2. Sign with `/usr/bin/openssl pkeyutl -sign` and assert byte-for-byte equality + pure-Rust verify!
    let sign_status = Command::new("/usr/bin/openssl")
        .args(["pkeyutl", "-sign", "-inkey"])
        .arg(&sk_der_path)
        .args(["-keyform", "DER", "-rawin", "-in"])
        .arg(&msg_path)
        .arg("-out")
        .arg(&ossl_sig_path)
        .status()
        .expect("invoke /usr/bin/openssl pkeyutl -sign");
    assert!(sign_status.success(), "OpenSSL pkeyutl -sign failed");
    let ossl_sig_bytes = fs::read(&ossl_sig_path).expect("read ossl sig");
    assert_eq!(ossl_sig_bytes.as_slice(), &rust_sig.0[..]);

    let mut ossl_sig = [0u8; 64];
    ossl_sig.copy_from_slice(&ossl_sig_bytes);
    assert!(ed25519_verify(&kp.public_key, msg, &ossl_sig));

    let _ = fs::remove_dir_all(&scratch);
}

#[test]
fn offline_macaroon_chain_and_signed_fence_protocol_unit_checks() {
    let issuers: Vec<Ed25519Keypair> = (0..3)
        .map(|i| Ed25519Keypair::deterministic("unit.issuer", i))
        .collect();
    let sink_kp = Ed25519Keypair::deterministic("unit.sink0", 0);
    let issuer_pks: Vec<[u8; 32]> = issuers.iter().map(|k| k.public_key).collect();
    let mac_key = sha256(b"unit.macaroon.key");

    let mut sink = CryptoSinkVerifier::new("sink-0", sink_kp.clone(), issuer_pks, mac_key);

    // Mint root macaroon for Alice (budget=10, scope=[T,T,T], req=[T,F,F])
    let root = CapabilityMacaroon::mint(
        &mac_key,
        MacaroonRoot {
            root_id: "cap-0".into(),
            sink_id: "sink-0".into(),
            initial_holder: "alice".into(),
            budget: 10,
            expires: 500,
            scope: vec![true, true, true],
            required: vec![true, false, false],
            epochs: vec![0, 0, 0],
        },
    )
    .expect("mint root");

    // Alice offline-attenuates to Bob (narrows budget 10->6, scope->[T,F,F], strengthens req->[T,T,F])
    let bob_cap = root
        .attenuate_offline(MacaroonCaveat {
            delegatee: "bob".into(),
            budget: 6,
            expires: 400,
            scope: vec![true, false, false],
            required: vec![true, true, false],
            pinned_proposal_digest: None,
        })
        .expect("alice -> bob");

    // Bob offline-attenuates to Carol (narrows budget 6->3, expires 400->300)
    let carol_prop = CanonicalProposal {
        id: "carol-tx".into(),
        actor: "carol".into(),
        sink_id: "sink-0".into(),
        epochs: vec![0, 0, 0],
        expires: 250,
        effect: CanonicalEffect::Transfer {
            recipient: "approved".into(),
            units: 3,
        },
    };
    let carol_cap = bob_cap
        .attenuate_offline(MacaroonCaveat {
            delegatee: "carol".into(),
            budget: 3,
            expires: 300,
            scope: vec![true, false, false],
            required: vec![true, true, false],
            pinned_proposal_digest: Some(carol_prop.digest()),
        })
        .expect("bob -> carol");

    assert!(
        sink.verify_macaroon_authorization(&carol_prop, "carol", &carol_cap, 100)
            .is_ok()
    );

    // Strengthening required=[T,T,F] means fencing issuer 1 MUST cancel Carol's transfer macaroon!
    let fence_i1 = SignedFenceCert::sign(&issuers[1], "sink-0", 1, 1);
    let ack = sink
        .apply_signed_fence(&fence_i1, 777)
        .expect("apply signed fence");
    assert!(ack.verify("sink-0", &sink_kp.public_key, 1, 1, 777));
    assert!(
        sink.verify_macaroon_authorization(&carol_prop, "carol", &carol_cap, 100)
            .is_err()
    );

    // QuorumCertificate sanity check
    let key_refs: Vec<Option<&Ed25519Keypair>> = issuers.iter().map(Some).collect();
    let rel_prop = CanonicalProposal {
        id: "rel-epoch1".into(),
        actor: "alice".into(),
        sink_id: "sink-0".into(),
        epochs: vec![0, 1, 0],
        expires: 400,
        effect: CanonicalEffect::Release {
            destination: "approved".into(),
            contents: "SYNTHETIC DATA: no personal information".into(),
        },
    };
    let qc = QuorumCertificate::sign_for_proposal(&rel_prop, vec![true, true, true], &key_refs);
    assert!(
        sink.verify_quorum_authorization(&rel_prop, "alice", &qc, 100)
            .is_ok()
    );
}

#[test]
fn redteam_and_jepsen_schedules_generate_verified_v4_evidence() {
    let quick = cfg!(debug_assertions);
    let expected_attacks = if quick { 1_000 } else { 10_000 };
    let expected_schedules = if quick { 200 } else { 1_000 };
    let min_per_family = if quick { 65 } else { 900 };

    let out_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-evidence")
        .join(format!("crypto-jepsen-v4-{}", std::process::id()))
        .join("result.json");
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).expect("mkdir test-evidence");
    }

    let evidence = crypto_and_jepsen_suite(20_261_110, quick);
    let mut bytes = serde_json::to_vec_pretty(&evidence).expect("serialize");
    bytes.push(b'\n');
    fs::write(&out_path, bytes).expect("write test evidence");

    // 1. Cryptographic red-team assertions: 0 false accepts, 100% positive controls
    assert_eq!(evidence.crypto_redteam.total_attacks, expected_attacks);
    assert_eq!(evidence.crypto_redteam.false_accepts, 0);
    assert_eq!(
        evidence.crypto_redteam.positive_controls_passed,
        expected_attacks
    );
    assert_eq!(evidence.crypto_redteam.families.len(), 10);
    for fam in &evidence.crypto_redteam.families {
        assert!(
            fam.attempts >= min_per_family,
            "under-sampled family {}: {}",
            fam.family,
            fam.attempts
        );
        assert_eq!(fam.false_accepts, 0, "false accept in {}", fam.family);
    }

    // 2. Multi-sink Jepsen campaign assertions across all 5 classes (including bounded clock skew)
    assert_eq!(evidence.jepsen_classes.len(), 5);
    let coop = &evidence.jepsen_classes[0];
    let premature = &evidence.jepsen_classes[1];
    let external = &evidence.jepsen_classes[2];
    let unbuf_skew = &evidence.jepsen_classes[3];
    let skew_buffered = &evidence.jepsen_classes[4];

    assert_eq!(coop.class, "cooperative_signed_fence");
    assert_eq!(coop.schedules_run, expected_schedules);
    assert_eq!(coop.post_ack_unauthorized_commits, 0);
    assert_eq!(coop.global_budget_breaches, 0);
    assert_eq!(coop.partition_invariant_breaches, 0);
    assert_eq!(coop.duplicate_effect_breaches, 0);
    assert_eq!(
        coop.mitm_forged_acks_injected,
        coop.mitm_forged_acks_rejected
    );
    assert!(coop.mitm_forged_acks_injected > expected_schedules / 10);
    assert!(coop.signed_fence_completions > 0);
    assert!(coop.lease_expiry_completions > 0);

    assert_eq!(premature.class, "premature_broker_only_ack");
    assert_eq!(premature.schedules_run, expected_schedules);
    assert!(
        premature.post_ack_unauthorized_commits > 0,
        "PrematureBrokerOnlyAck ablation must expose post-ack execution leaks under asymmetric partition"
    );
    assert_eq!(premature.global_budget_breaches, 0);
    assert_eq!(premature.duplicate_effect_breaches, 0);

    assert_eq!(external.class, "uncooperative_external_sink");
    assert_eq!(external.schedules_run, expected_schedules);
    assert!(
        external.post_ack_unauthorized_commits > 0,
        "UncooperativeExternalSink ablation must expose irrevocable in-flight external leaks"
    );
    assert_eq!(external.global_budget_breaches, 0);
    assert_eq!(external.duplicate_effect_breaches, 0);

    assert_eq!(unbuf_skew.class, "unbuffered_clock_skew");
    assert_eq!(unbuf_skew.schedules_run, expected_schedules);
    assert!(
        unbuf_skew.post_ack_unauthorized_commits > 0,
        "UnbufferedClockSkew negative control must expose post-ack leaks when partitioned shard clocks lag"
    );
    assert_eq!(unbuf_skew.global_budget_breaches, 0);
    assert_eq!(unbuf_skew.duplicate_effect_breaches, 0);

    assert_eq!(skew_buffered.class, "skew_buffered_cooperative_fence");
    assert_eq!(skew_buffered.schedules_run, expected_schedules);
    assert_eq!(
        skew_buffered.post_ack_unauthorized_commits, 0,
        "SkewBufferedCooperativeFence must eliminate all post-ack leaks under bounded clock skew"
    );
    assert_eq!(skew_buffered.global_budget_breaches, 0);
    assert_eq!(skew_buffered.partition_invariant_breaches, 0);
    assert_eq!(skew_buffered.duplicate_effect_breaches, 0);

    assert!(
        out_path.is_file(),
        "missing artifact {}",
        out_path.display()
    );
}
