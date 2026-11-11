//! Cryptographic capability macaroons, multi-issuer Ed25519 quorum certificates,
//! and signed monotone epoch-fence / fence-ack certificates.
//!
//! Pure-Rust zero-external-crypto-crate implementation of:
//! - FIPS 180-4 SHA-256 and SHA-512
//! - RFC 2104 / RFC 4231 HMAC-SHA256
//! - RFC 8032 PureEdDSA Ed25519 over twisted Edwards curve -x^2 + y^2 = 1 + d x^2 y^2
//!   modulo 2^255 - 19, with strict canonical scalar S < L and canonical point checks
//! - Domain-separated length-prefixed canonical framing (`jeath.*.v4`)
//! - Multi-issuer Ed25519 quorum certificates (`jeath.quorum.v4`)
//! - Offline-attenuable HMAC-SHA256 capability macaroons (`jeath.macaroon.*.v4`)
//!   enforcing monotone attenuation via `policy_core::attenuates`
//! - Signed monotone epoch-fence certificates (`jeath.fence.v4`) and signed sink
//!   fence-acknowledgment receipts (`jeath.fence_ack.v4`)

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub const DOMAIN_PROPOSAL_V4: &[u8] = b"jeath.proposal.v4";
pub const DOMAIN_QUORUM_V4: &[u8] = b"jeath.quorum.v4";
pub const DOMAIN_MACAROON_ROOT_V4: &[u8] = b"jeath.macaroon.root.v4";
pub const DOMAIN_MACAROON_CAVEAT_V4: &[u8] = b"jeath.macaroon.caveat.v4";
pub const DOMAIN_FENCE_V4: &[u8] = b"jeath.fence.v4";
pub const DOMAIN_FENCE_ACK_V4: &[u8] = b"jeath.fence_ack.v4";
pub const DOMAIN_LEASE_V4: &[u8] = b"jeath.lease.v4";

// ============================================================================
// Constant-time byte comparison & hex helpers
// ============================================================================

#[inline]
pub fn ct_eq_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    let mut i = 0;
    while i < 32 {
        diff |= a[i] ^ b[i];
        i += 1;
    }
    diff == 0
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

pub fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex byte 0x{b:02x}")),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sig64(pub [u8; 64]);

impl Serialize for Sig64 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&hex_encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for Sig64 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let raw = hex_decode(&s).map_err(serde::de::Error::custom)?;
        if raw.len() != 64 {
            return Err(serde::de::Error::custom("expected 64-byte signature hex"));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&raw);
        Ok(Self(arr))
    }
}

// ============================================================================
// FIPS 180-4 SHA-256 & RFC 2104 / RFC 4231 HMAC-SHA256
// ============================================================================

const H256_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
pub struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self {
            h: H256_INIT,
            buf: [0u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                Self::compress(&mut self.h, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress(&mut self.h, &block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            Self::compress(&mut self.h, &block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.h, &block);

        let mut out = [0u8; 32];
        for (i, &word) in self.h.iter().enumerate() {
            out[4 * i..4 * (i + 1)].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[4 * i],
                block[4 * i + 1],
                block[4 * i + 2],
                block[4 * i + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut k_block = [0u8; 64];
    if key.len() > 64 {
        let kh = sha256(key);
        k_block[..32].copy_from_slice(&kh);
    } else {
        k_block[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k_block[i];
        opad[i] ^= k_block[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(data);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    outer.finalize()
}

// ============================================================================
// FIPS 180-4 SHA-512 (for RFC 8032 Ed25519)
// ============================================================================

const H512_INIT: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const K512: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

#[derive(Clone)]
pub struct Sha512 {
    h: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    total_bytes: u128,
}

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha512 {
    pub fn new() -> Self {
        Self {
            h: H512_INIT,
            buf: [0u8; 128],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u128);
        if self.buf_len > 0 {
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 128 {
                let block = self.buf;
                Self::compress(&mut self.h, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 128 {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[..128]);
            Self::compress(&mut self.h, &block);
            data = &data[128..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 64] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 112 {
            self.buf[self.buf_len..128].fill(0);
            let block = self.buf;
            Self::compress(&mut self.h, &block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..112].fill(0);
        self.buf[112..128].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.h, &block);

        let mut out = [0u8; 64];
        for (i, &word) in self.h.iter().enumerate() {
            out[8 * i..8 * (i + 1)].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(state: &mut [u64; 8], block: &[u8; 128]) {
        let mut w = [0u64; 80];
        for i in 0..16 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&block[8 * i..8 * (i + 1)]);
            w[i] = u64::from_be_bytes(bytes);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..80 {
            let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K512[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }
}

pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut hasher = Sha512::new();
    hasher.update(data);
    hasher.finalize()
}

// ============================================================================
// RFC 8032 Pure-Rust Ed25519 (GF(2^255 - 19) + Extended Twisted Edwards)
// ============================================================================

const MASK51: u64 = (1u64 << 51) - 1;

#[derive(Clone, Copy, Debug)]
struct Fe([u64; 5]);

const FE_ZERO: Fe = Fe([0, 0, 0, 0, 0]);
const FE_ONE: Fe = Fe([1, 0, 0, 0, 0]);

// d = -121665 / 121666 mod (2^255 - 19)
const FE_D: Fe = Fe([
    0x34dca135978a3,
    0x1a8283b156ebd,
    0x5e7a26001c029,
    0x739c663a03cbb,
    0x52036cee2b6ff,
]);

// 2 * d mod (2^255 - 19)
const FE_2D: Fe = Fe([
    0x69b9426b2f159,
    0x35050762add7a,
    0x3cf44c0038052,
    0x6738cc7407977,
    0x2406d9dc56dff,
]);

// sqrt(-1) = 2^((p - 1) / 4) mod p
const FE_SQRT_M1: Fe = Fe([
    0x61b274a0ea0b0,
    0x0d5a5fc8f189d,
    0x7ef5e9cbd0c60,
    0x78595a6804c9e,
    0x2b8324804fc1d,
]);

const FE_BX: Fe = Fe([
    0x62d608f25d51a,
    0x412a4b4f6592a,
    0x75b7171a4b31d,
    0x1ff60527118fe,
    0x216936d3cd6e5,
]);

const FE_BY: Fe = Fe([
    0x6666666666658,
    0x4cccccccccccc,
    0x1999999999999,
    0x3333333333333,
    0x6666666666666,
]);

impl Fe {
    #[inline]
    fn carry(mut h: [u64; 5]) -> Self {
        for _ in 0..2 {
            let mut i = 0;
            while i < 4 {
                let c = h[i] >> 51;
                h[i] &= MASK51;
                h[i + 1] += c;
                i += 1;
            }
            let c4 = h[4] >> 51;
            h[4] &= MASK51;
            h[0] += c4 * 19;
        }
        Self(h)
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::carry([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
            self.0[3] + rhs.0[3],
            self.0[4] + rhs.0[4],
        ])
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        // Add 4*p in radix-2^51 so each limb stays strictly positive before subtraction.
        const FOUR_P0: u64 = 4 * (MASK51 - 18);
        const FOUR_PK: u64 = 4 * MASK51;
        Self::carry([
            (self.0[0] + FOUR_P0) - rhs.0[0],
            (self.0[1] + FOUR_PK) - rhs.0[1],
            (self.0[2] + FOUR_PK) - rhs.0[2],
            (self.0[3] + FOUR_PK) - rhs.0[3],
            (self.0[4] + FOUR_PK) - rhs.0[4],
        ])
    }

    #[inline]
    fn neg(self) -> Self {
        FE_ZERO.sub(self)
    }

    fn mul(self, rhs: Self) -> Self {
        let a = [
            self.0[0] as u128,
            self.0[1] as u128,
            self.0[2] as u128,
            self.0[3] as u128,
            self.0[4] as u128,
        ];
        let b = [
            rhs.0[0] as u128,
            rhs.0[1] as u128,
            rhs.0[2] as u128,
            rhs.0[3] as u128,
            rhs.0[4] as u128,
        ];
        let b19 = [0, b[1] * 19, b[2] * 19, b[3] * 19, b[4] * 19];

        let mut t0 = a[0] * b[0] + a[1] * b19[4] + a[2] * b19[3] + a[3] * b19[2] + a[4] * b19[1];
        let mut t1 = a[0] * b[1] + a[1] * b[0] + a[2] * b19[4] + a[3] * b19[3] + a[4] * b19[2];
        let mut t2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + a[3] * b19[4] + a[4] * b19[3];
        let mut t3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0] + a[4] * b19[4];
        let mut t4 = a[0] * b[4] + a[1] * b[3] + a[2] * b[2] + a[3] * b[1] + a[4] * b[0];

        let m = MASK51 as u128;
        let c0 = t0 >> 51;
        t0 &= m;
        t1 += c0;
        let c1 = t1 >> 51;
        t1 &= m;
        t2 += c1;
        let c2 = t2 >> 51;
        t2 &= m;
        t3 += c2;
        let c3 = t3 >> 51;
        t3 &= m;
        t4 += c3;
        let c4 = t4 >> 51;
        t4 &= m;
        t0 += c4 * 19;
        let c0b = t0 >> 51;
        t0 &= m;
        t1 += c0b;

        Self::carry([t0 as u64, t1 as u64, t2 as u64, t3 as u64, t4 as u64])
    }

    #[inline]
    fn sq(self) -> Self {
        self.mul(self)
    }

    /// Canonical reduction modulo p = 2^255 - 19 into 51-bit limbs.
    fn freeze(self) -> [u64; 5] {
        let mut r = Self::carry(self.0).0;
        // Subtract p = (2^51 - 19) + sum_{k=1..4} (2^51 - 1) * 2^(51*k) if r >= p.
        let mut q = (r[0] + 19) >> 51;
        q = (r[1] + q) >> 51;
        q = (r[2] + q) >> 51;
        q = (r[3] + q) >> 51;
        q = (r[4] + q) >> 51;
        r[0] += 19 * q;
        let c0 = r[0] >> 51;
        r[0] &= MASK51;
        r[1] += c0;
        let c1 = r[1] >> 51;
        r[1] &= MASK51;
        r[2] += c1;
        let c2 = r[2] >> 51;
        r[2] &= MASK51;
        r[3] += c2;
        let c3 = r[3] >> 51;
        r[3] &= MASK51;
        r[4] = (r[4] + c3) & MASK51;
        r
    }

    fn to_bytes(self) -> [u8; 32] {
        let l = self.freeze();
        let mut acc = 0u128;
        let mut bits = 0u32;
        let mut out = [0u8; 32];
        let mut idx = 0usize;
        for &limb in &l {
            acc |= (limb as u128) << bits;
            bits += 51;
            while bits >= 8 && idx < 32 {
                out[idx] = (acc & 0xff) as u8;
                acc >>= 8;
                bits -= 8;
                idx += 1;
            }
        }
        if idx < 32 {
            out[idx] = (acc & 0xff) as u8;
        }
        out
    }

    /// Strict RFC 8032 §5.1.3 canonical field element decoding (clears bit 255 first, rejects >= p).
    fn from_canonical_bytes(bytes: &[u8; 32]) -> Option<Self> {
        let mut clean = *bytes;
        clean[31] &= 0x7f;

        let mut limbs = [0u64; 5];
        let mut bit_pos = 0usize;
        for slot in &mut limbs {
            let byte_idx = bit_pos / 8;
            let bit_off = bit_pos % 8;
            let mut raw = 0u128;
            let max_bytes = (32 - byte_idx).min(8);
            for k in 0..max_bytes {
                raw |= (clean[byte_idx + k] as u128) << (8 * k);
            }
            *slot = ((raw >> bit_off) as u64) & MASK51;
            bit_pos += 51;
        }
        let fe = Self(limbs);
        if !ct_eq_32(&fe.to_bytes(), &clean) {
            return None;
        }
        Some(fe)
    }

    fn is_zero(self) -> bool {
        ct_eq_32(&self.to_bytes(), &[0u8; 32])
    }

    fn parity(self) -> u8 {
        self.to_bytes()[0] & 1
    }

    /// Modular exponentiation x^exp mod p where exp is given as 4 little-endian u64 words.
    fn pow_u256(self, exp: [u64; 4]) -> Self {
        let mut res = FE_ONE;
        let mut base = self;
        for &word in &exp {
            let mut w = word;
            for _ in 0..64 {
                if (w & 1) == 1 {
                    res = res.mul(base);
                }
                base = base.sq();
                w >>= 1;
            }
        }
        res
    }

    fn invert(self) -> Self {
        // p - 2 = 2^255 - 21
        self.pow_u256([
            0xffffffffffffffeb,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x7fffffffffffffff,
        ])
    }

    fn pow_p_plus_3_div_8(self) -> Self {
        // (p + 3) / 8 = (2^255 - 16) / 8 = 2^252 - 2
        self.pow_u256([
            0xfffffffffffffffe,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x0fffffffffffffff,
        ])
    }
}

#[derive(Clone, Copy, Debug)]
struct ExtPoint {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl ExtPoint {
    fn identity() -> Self {
        Self {
            x: FE_ZERO,
            y: FE_ONE,
            z: FE_ONE,
            t: FE_ZERO,
        }
    }

    fn basepoint() -> Self {
        Self {
            x: FE_BX,
            y: FE_BY,
            z: FE_ONE,
            t: FE_BX.mul(FE_BY),
        }
    }

    /// Complete unified twisted Edwards addition on -x^2 + y^2 = 1 + d x^2 y^2 (RFC 8032 §5.1.4).
    fn add(self, other: Self) -> Self {
        let a = self.y.sub(self.x).mul(other.y.sub(other.x));
        let b = self.y.add(self.x).mul(other.y.add(other.x));
        let c = self.t.mul(FE_2D).mul(other.t);
        let d = self.z.add(self.z).mul(other.z);
        let e = b.sub(a);
        let f = d.sub(c);
        let g = d.add(c);
        let h = b.add(a);
        Self {
            x: e.mul(f),
            y: g.mul(h),
            z: f.mul(g),
            t: e.mul(h),
        }
    }

    fn double(self) -> Self {
        self.add(self)
    }

    fn scalar_mul(self, scalar_le: &[u8; 32]) -> Self {
        let mut win = [Self::identity(); 16];
        win[1] = self;
        for i in 2..16 {
            win[i] = win[i - 1].add(self);
        }
        let mut acc = Self::identity();
        let mut byte_idx = 32;
        while byte_idx > 0 {
            byte_idx -= 1;
            let b = scalar_le[byte_idx];
            let hi = (b >> 4) as usize;
            acc = acc.double().double().double().double();
            if hi != 0 {
                acc = acc.add(win[hi]);
            }
            let lo = (b & 0x0f) as usize;
            acc = acc.double().double().double().double();
            if lo != 0 {
                acc = acc.add(win[lo]);
            }
        }
        acc
    }

    fn basepoint_mul(scalar_le: &[u8; 32]) -> Self {
        static BASE_TABLE: std::sync::OnceLock<Box<[[ExtPoint; 16]; 64]>> =
            std::sync::OnceLock::new();
        let table = BASE_TABLE.get_or_init(|| {
            let mut tbl = Box::new([[ExtPoint::identity(); 16]; 64]);
            let mut radix_base = ExtPoint::basepoint();
            for slot in tbl.iter_mut() {
                slot[1] = radix_base;
                for j in 2..16 {
                    slot[j] = slot[j - 1].add(radix_base);
                }
                // Advance radix_base by 16 * radix_base for next 4-bit nibble
                radix_base = slot[8].double();
            }
            tbl
        });
        let mut acc = Self::identity();
        for (byte_i, &b) in scalar_le.iter().enumerate() {
            let lo = (b & 0x0f) as usize;
            if lo != 0 {
                acc = acc.add(table[2 * byte_i][lo]);
            }
            let hi = (b >> 4) as usize;
            if hi != 0 {
                acc = acc.add(table[2 * byte_i + 1][hi]);
            }
        }
        acc
    }

    fn to_bytes(self) -> [u8; 32] {
        let z_inv = self.z.invert();
        let x = self.x.mul(z_inv);
        let y = self.y.mul(z_inv);
        let mut out = y.to_bytes();
        out[31] |= x.parity() << 7;
        out
    }

    fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        let x_0 = (bytes[31] >> 7) & 1;
        let y = Fe::from_canonical_bytes(bytes)?;
        let y2 = y.sq();
        let u = y2.sub(FE_ONE);
        let v = FE_D.mul(y2).add(FE_ONE);
        let w = u.mul(v.invert());
        let mut x = w.pow_p_plus_3_div_8();
        let check = v.mul(x.sq());
        if !ct_eq_32(&check.to_bytes(), &u.to_bytes()) {
            let neg_u = u.neg();
            if ct_eq_32(&check.to_bytes(), &neg_u.to_bytes()) {
                x = x.mul(FE_SQRT_M1);
            } else {
                return None;
            }
        }
        if x.is_zero() && x_0 == 1 {
            return None;
        }
        if x.parity() != x_0 {
            x = x.neg();
        }
        Some(Self {
            x,
            y,
            z: FE_ONE,
            t: x.mul(y),
        })
    }
}

// Group order L = 2^252 + 27742317777372353535851937790883648493
const SCALAR_L: [u64; 5] = [
    0x5812631a5cf5d3ed,
    0x14def9dea2f79cd6,
    0x0000000000000000,
    0x1000000000000000,
    0x0000000000000000,
];

fn u320_ge_l(x: &[u64; 5]) -> bool {
    let mut i = 5;
    while i > 0 {
        i -= 1;
        if x[i] != SCALAR_L[i] {
            return x[i] > SCALAR_L[i];
        }
    }
    true
}

fn u320_sub_l(x: &mut [u64; 5]) {
    let mut borrow = 0u128;
    for i in 0..5 {
        let lhs = x[i] as u128;
        let rhs = (SCALAR_L[i] as u128) + borrow;
        if lhs >= rhs {
            x[i] = (lhs - rhs) as u64;
            borrow = 0;
        } else {
            x[i] = ((1u128 << 64) + lhs - rhs) as u64;
            borrow = 1;
        }
    }
}

fn reduce_words_mod_l(words: &[u64]) -> [u8; 32] {
    let mut rem = [0u64; 5];
    let total_bits = words.len() * 64;
    let mut bit = total_bits;
    while bit > 0 {
        bit -= 1;
        let in_bit = (words[bit / 64] >> (bit % 64)) & 1;
        let mut carry = in_bit;
        for w in &mut rem {
            let next_carry = *w >> 63;
            *w = (*w << 1) | carry;
            carry = next_carry;
        }
        if u320_ge_l(&rem) {
            u320_sub_l(&mut rem);
        }
    }
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[8 * i..8 * (i + 1)].copy_from_slice(&rem[i].to_le_bytes());
    }
    out
}

pub fn scalar_reduce_512(hash: &[u8; 64]) -> [u8; 32] {
    let mut words = [0u64; 8];
    for i in 0..8 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&hash[8 * i..8 * (i + 1)]);
        words[i] = u64::from_le_bytes(b);
    }
    reduce_words_mod_l(&words)
}

/// Computes (a * b + c) mod L where a, b, c are 256-bit little-endian integers.
fn scalar_mul_add(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> [u8; 32] {
    let to_u64s = |bytes: &[u8; 32]| -> [u64; 4] {
        let mut w = [0u64; 4];
        for i in 0..4 {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&bytes[8 * i..8 * (i + 1)]);
            w[i] = u64::from_le_bytes(chunk);
        }
        w
    };
    let aw = to_u64s(a);
    let bw = to_u64s(b);
    let cw = to_u64s(c);

    let mut acc = [0u64; 9];
    for (i, &cv) in cw.iter().enumerate() {
        acc[i] = cv;
    }
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let prod = (aw[i] as u128) * (bw[j] as u128) + (acc[i + j] as u128) + carry;
            acc[i + j] = prod as u64;
            carry = prod >> 64;
        }
        let mut k = i + 4;
        while carry > 0 {
            let sum = (acc[k] as u128) + carry;
            acc[k] = sum as u64;
            carry = sum >> 64;
            k += 1;
        }
    }
    reduce_words_mod_l(&acc)
}

pub fn scalar_is_canonical(s: &[u8; 32]) -> bool {
    let mut w = [0u64; 5];
    for i in 0..4 {
        let mut chunk = [0u8; 8];
        chunk.copy_from_slice(&s[8 * i..8 * (i + 1)]);
        w[i] = u64::from_le_bytes(chunk);
    }
    !u320_ge_l(&w)
}

#[derive(Clone, Debug)]
pub struct Ed25519Keypair {
    pub seed: [u8; 32],
    pub public_key: [u8; 32],
    secret_scalar: [u8; 32],
    prefix: [u8; 32],
}

impl Ed25519Keypair {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let h = sha512(&seed);
        let mut secret_scalar = [0u8; 32];
        secret_scalar.copy_from_slice(&h[..32]);
        secret_scalar[0] &= 248;
        secret_scalar[31] &= 127;
        secret_scalar[31] |= 64;
        let mut prefix = [0u8; 32];
        prefix.copy_from_slice(&h[32..64]);
        let public_key = ExtPoint::basepoint_mul(&secret_scalar).to_bytes();
        Self {
            seed,
            public_key,
            secret_scalar,
            prefix,
        }
    }

    pub fn deterministic(domain_label: &str, index: u64) -> Self {
        let mut w = CanonicalWriter::with_domain(b"jeath.keygen.v4");
        w.write_str(domain_label);
        w.write_u64(index);
        Self::from_seed(sha256(&w.into_bytes()))
    }

    pub fn sign(&self, message: &[u8]) -> Sig64 {
        let mut r_hasher = Sha512::new();
        r_hasher.update(&self.prefix);
        r_hasher.update(message);
        let r = scalar_reduce_512(&r_hasher.finalize());
        let r_bytes = ExtPoint::basepoint_mul(&r).to_bytes();

        let mut k_hasher = Sha512::new();
        k_hasher.update(&r_bytes);
        k_hasher.update(&self.public_key);
        k_hasher.update(message);
        let k = scalar_reduce_512(&k_hasher.finalize());

        let s = scalar_mul_add(&k, &self.secret_scalar, &r);
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&r_bytes);
        sig[32..].copy_from_slice(&s);
        Sig64(sig)
    }
}

pub fn ed25519_public_key(seed: &[u8; 32]) -> [u8; 32] {
    Ed25519Keypair::from_seed(*seed).public_key
}

pub fn ed25519_sign(seed: &[u8; 32], message: &[u8]) -> [u8; 64] {
    Ed25519Keypair::from_seed(*seed).sign(message).0
}

pub fn ed25519_verify(public_key: &[u8; 32], message: &[u8], signature: &[u8; 64]) -> bool {
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&signature[..32]);
    s_bytes.copy_from_slice(&signature[32..]);

    if !scalar_is_canonical(&s_bytes) {
        return false;
    }
    let Some(a_point) = ExtPoint::from_bytes(public_key) else {
        return false;
    };
    let Some(r_point) = ExtPoint::from_bytes(&r_bytes) else {
        return false;
    };

    let mut k_hasher = Sha512::new();
    k_hasher.update(&r_bytes);
    k_hasher.update(public_key);
    k_hasher.update(message);
    let k = scalar_reduce_512(&k_hasher.finalize());

    // Exact RFC 8032 §5.1.7 verification equation: [S]B == R + [k]A
    let lhs = ExtPoint::basepoint_mul(&s_bytes).to_bytes();
    let rhs = r_point.add(a_point.scalar_mul(&k)).to_bytes();
    ct_eq_32(&lhs, &rhs)
}

// ============================================================================
// Domain-Separated Length-Prefixed Canonical Framing
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct CanonicalWriter {
    buf: Vec<u8>,
}

impl CanonicalWriter {
    pub fn with_domain(domain: &[u8]) -> Self {
        let mut w = Self {
            buf: Vec::with_capacity(128),
        };
        w.write_bytes(domain);
        w
    }

    #[inline]
    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline]
    pub fn write_bool(&mut self, v: bool) {
        self.buf.push(u8::from(v));
    }

    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    pub fn write_bytes(&mut self, b: &[u8]) {
        self.write_u64(b.len() as u64);
        self.buf.extend_from_slice(b);
    }

    #[inline]
    pub fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    #[inline]
    pub fn write_bool_slice(&mut self, xs: &[bool]) {
        self.write_u64(xs.len() as u64);
        for &b in xs {
            self.write_bool(b);
        }
    }

    #[inline]
    pub fn write_u64_slice(&mut self, xs: &[u64]) {
        self.write_u64(xs.len() as u64);
        for &x in xs {
            self.write_u64(x);
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

// ============================================================================
// Canonical Effect / Proposal Model & Digest
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanonicalEffect {
    Transfer {
        recipient: String,
        units: u64,
    },
    Release {
        destination: String,
        contents: String,
    },
    Edit {
        document: String,
        expected_revision: u64,
        contents: String,
    },
}

impl CanonicalEffect {
    pub fn charge(&self) -> u64 {
        match self {
            Self::Transfer { units, .. } => *units,
            Self::Release { .. } | Self::Edit { .. } => 1,
        }
    }

    pub fn kind_index(&self) -> usize {
        match self {
            Self::Transfer { .. } => 0,
            Self::Release { .. } => 1,
            Self::Edit { .. } => 2,
        }
    }

    pub fn base_required(&self, issuers: usize) -> Vec<bool> {
        let mut req = vec![false; issuers];
        if issuers == 0 {
            return req;
        }
        match self {
            Self::Transfer { .. } => {
                req[0] = true;
            }
            Self::Edit { .. } => {
                req[0] = true;
                if issuers > 1 {
                    req[1] = true;
                }
            }
            Self::Release { .. } => {
                req.fill(true);
            }
        }
        req
    }

    pub fn scoped(&self) -> bool {
        match self {
            Self::Transfer { recipient, units } => recipient == "approved" && *units > 0,
            Self::Release {
                destination,
                contents,
            } => destination == "approved" && contents == "SYNTHETIC DATA: no personal information",
            Self::Edit {
                document, contents, ..
            } => document == "toy.rs" && contents.len() <= 4096,
        }
    }

    pub fn write_canonical(&self, w: &mut CanonicalWriter) {
        match self {
            Self::Transfer { recipient, units } => {
                w.write_u8(0x00);
                w.write_str(recipient);
                w.write_u64(*units);
            }
            Self::Release {
                destination,
                contents,
            } => {
                w.write_u8(0x01);
                w.write_str(destination);
                w.write_str(contents);
            }
            Self::Edit {
                document,
                expected_revision,
                contents,
            } => {
                w.write_u8(0x02);
                w.write_str(document);
                w.write_u64(*expected_revision);
                w.write_str(contents);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CanonicalProposal {
    pub id: String,
    pub actor: String,
    pub sink_id: String,
    pub epochs: Vec<u64>,
    pub expires: u64,
    pub effect: CanonicalEffect,
}

impl CanonicalProposal {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut w = CanonicalWriter::with_domain(DOMAIN_PROPOSAL_V4);
        w.write_str(&self.id);
        w.write_str(&self.actor);
        w.write_str(&self.sink_id);
        w.write_u64_slice(&self.epochs);
        w.write_u64(self.expires);
        self.effect.write_canonical(&mut w);
        w.into_bytes()
    }

    pub fn digest(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

// ============================================================================
// Multi-Issuer Ed25519 Quorum Certificates (`jeath.quorum.v4`)
// ============================================================================

pub fn quorum_vote_frame(
    sink_id: &str,
    proposal_digest: &[u8; 32],
    issuer: usize,
    epoch: u64,
) -> Vec<u8> {
    let mut w = CanonicalWriter::with_domain(DOMAIN_QUORUM_V4);
    w.write_str(sink_id);
    w.write_bytes(proposal_digest);
    w.write_u64(issuer as u64);
    w.write_u64(epoch);
    w.into_bytes()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IssuerQuorumVote {
    pub issuer: usize,
    pub epoch: u64,
    pub signature: Sig64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuorumCertificate {
    pub sink_id: String,
    pub proposal_digest: [u8; 32],
    pub required: Vec<bool>,
    pub votes: Vec<IssuerQuorumVote>,
}

impl QuorumCertificate {
    pub fn sign_for_proposal(
        proposal: &CanonicalProposal,
        required: Vec<bool>,
        issuer_keys: &[Option<&Ed25519Keypair>],
    ) -> Self {
        let digest = proposal.digest();
        let mut votes = Vec::new();
        for (i, &need) in required.iter().enumerate() {
            if need
                && let Some(Some(kp)) = issuer_keys.get(i)
                && let Some(&ep) = proposal.epochs.get(i)
            {
                let frame = quorum_vote_frame(&proposal.sink_id, &digest, i, ep);
                votes.push(IssuerQuorumVote {
                    issuer: i,
                    epoch: ep,
                    signature: kp.sign(&frame),
                });
            }
        }
        Self {
            sink_id: proposal.sink_id.clone(),
            proposal_digest: digest,
            required,
            votes,
        }
    }
}

// ============================================================================
// Offline-Attenuable HMAC-SHA256 Capability Macaroons (`jeath.macaroon.*.v4`)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MacaroonRoot {
    pub root_id: String,
    pub sink_id: String,
    pub initial_holder: String,
    pub budget: u64,
    pub expires: u64,
    pub scope: Vec<bool>,
    pub required: Vec<bool>,
    pub epochs: Vec<u64>,
}

impl MacaroonRoot {
    pub fn canonical_frame(&self) -> Vec<u8> {
        let mut w = CanonicalWriter::with_domain(DOMAIN_MACAROON_ROOT_V4);
        w.write_str(&self.root_id);
        w.write_str(&self.sink_id);
        w.write_str(&self.initial_holder);
        w.write_u64(self.budget);
        w.write_u64(self.expires);
        w.write_bool_slice(&self.scope);
        w.write_bool_slice(&self.required);
        w.write_u64_slice(&self.epochs);
        w.into_bytes()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MacaroonCaveat {
    pub delegatee: String,
    pub budget: u64,
    pub expires: u64,
    pub scope: Vec<bool>,
    pub required: Vec<bool>,
    pub pinned_proposal_digest: Option<[u8; 32]>,
}

impl MacaroonCaveat {
    pub fn canonical_frame(&self, step_index: usize) -> Vec<u8> {
        let mut w = CanonicalWriter::with_domain(DOMAIN_MACAROON_CAVEAT_V4);
        w.write_u64(step_index as u64);
        w.write_str(&self.delegatee);
        w.write_u64(self.budget);
        w.write_u64(self.expires);
        w.write_bool_slice(&self.scope);
        w.write_bool_slice(&self.required);
        match &self.pinned_proposal_digest {
            None => w.write_u8(0),
            Some(d) => {
                w.write_u8(1);
                w.write_bytes(d);
            }
        }
        w.into_bytes()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityMacaroon {
    pub root: MacaroonRoot,
    pub caveats: Vec<MacaroonCaveat>,
    pub tag: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedCapability {
    pub holder: String,
    pub budget: u64,
    pub expires: u64,
    pub scope: Vec<bool>,
    pub required: Vec<bool>,
    pub epochs: Vec<u64>,
    pub pinned_proposal_digest: Option<[u8; 32]>,
}

impl CapabilityMacaroon {
    pub fn mint(broker_root_key: &[u8; 32], root: MacaroonRoot) -> Result<Self, String> {
        if root.budget == 0
            || root.scope.len() != 3
            || !root.scope.iter().any(|&b| b)
            || root.required.is_empty()
            || root.required.len() != root.epochs.len()
        {
            return Err("invalid macaroon root shape".into());
        }
        let tag = hmac_sha256(broker_root_key, &root.canonical_frame());
        Ok(Self {
            root,
            caveats: Vec::new(),
            tag,
        })
    }

    /// Offline attenuation by current holder without access to `broker_root_key`.
    /// Checks `policy_core::attenuates` locally; even a malicious holder who bypasses
    /// this check and chains `hmac_sha256` directly is rejected at verification time.
    pub fn attenuate_offline(&self, caveat: MacaroonCaveat) -> Result<Self, String> {
        let cur = self.effective_unchecked();
        if caveat.budget == 0
            || !policy_core::attenuates(
                cur.budget,
                caveat.budget,
                cur.expires,
                caveat.expires,
                &cur.scope,
                &caveat.scope,
                &cur.required,
                &caveat.required,
            )
            || !caveat.scope.iter().any(|&b| b)
        {
            return Err("caveat violates monotone attenuation".into());
        }
        if let Some(pinned) = cur.pinned_proposal_digest
            && caveat.pinned_proposal_digest != Some(pinned)
        {
            return Err("caveat cannot unpin or alter pinned proposal digest".into());
        }
        Ok(self.append_caveat_raw_for_test(caveat))
    }

    /// Raw HMAC chaining helper (used by red-team tests to simulate an attacker who
    /// legitimately knows `self.tag` and tries to chain an authority-inflating caveat).
    pub fn append_caveat_raw_for_test(&self, caveat: MacaroonCaveat) -> Self {
        let step_idx = self.caveats.len();
        let next_tag = hmac_sha256(&self.tag, &caveat.canonical_frame(step_idx));
        let mut caveats = self.caveats.clone();
        caveats.push(caveat);
        Self {
            root: self.root.clone(),
            caveats,
            tag: next_tag,
        }
    }

    fn effective_unchecked(&self) -> VerifiedCapability {
        if let Some(last) = self.caveats.last() {
            VerifiedCapability {
                holder: last.delegatee.clone(),
                budget: last.budget,
                expires: last.expires,
                scope: last.scope.clone(),
                required: last.required.clone(),
                epochs: self.root.epochs.clone(),
                pinned_proposal_digest: last.pinned_proposal_digest,
            }
        } else {
            VerifiedCapability {
                holder: self.root.initial_holder.clone(),
                budget: self.root.budget,
                expires: self.root.expires,
                scope: self.root.scope.clone(),
                required: self.root.required.clone(),
                epochs: self.root.epochs.clone(),
                pinned_proposal_digest: None,
            }
        }
    }

    pub fn verify(&self, broker_root_key: &[u8; 32]) -> Result<VerifiedCapability, String> {
        if self.root.budget == 0
            || self.root.scope.len() != 3
            || !self.root.scope.iter().any(|&b| b)
            || self.root.required.is_empty()
            || self.root.required.len() != self.root.epochs.len()
        {
            return Err("invalid root macaroon parameters".into());
        }
        let mut tag = hmac_sha256(broker_root_key, &self.root.canonical_frame());
        let mut holder = self.root.initial_holder.clone();
        let mut budget = self.root.budget;
        let mut expires = self.root.expires;
        let mut scope = self.root.scope.clone();
        let mut required = self.root.required.clone();
        let mut pinned: Option<[u8; 32]> = None;

        for (idx, c) in self.caveats.iter().enumerate() {
            if c.delegatee.is_empty() || c.delegatee.len() > 128 {
                return Err("invalid delegatee identifier".into());
            }
            if c.budget == 0 || !c.scope.iter().any(|&b| b) {
                return Err("empty caveat budget or scope".into());
            }
            if !policy_core::attenuates(
                budget,
                c.budget,
                expires,
                c.expires,
                &scope,
                &c.scope,
                &required,
                &c.required,
            ) {
                return Err("caveat violates monotone policy_core::attenuates".into());
            }
            if let Some(prev_digest) = pinned
                && c.pinned_proposal_digest != Some(prev_digest)
            {
                return Err("caveat mutated pinned proposal digest".into());
            }
            tag = hmac_sha256(&tag, &c.canonical_frame(idx));
            holder = c.delegatee.clone();
            budget = c.budget;
            expires = c.expires;
            scope = c.scope.clone();
            required = c.required.clone();
            pinned = c.pinned_proposal_digest;
        }

        if !ct_eq_32(&tag, &self.tag) {
            return Err("macaroon HMAC tag mismatch".into());
        }
        Ok(VerifiedCapability {
            holder,
            budget,
            expires,
            scope,
            required,
            epochs: self.root.epochs.clone(),
            pinned_proposal_digest: pinned,
        })
    }
}

// ============================================================================
// Signed Monotone Epoch-Fence Certificates & Signed Sink FenceAcks
// ============================================================================

pub fn fence_cert_frame(sink_id: &str, issuer: usize, floor: u64) -> Vec<u8> {
    let mut w = CanonicalWriter::with_domain(DOMAIN_FENCE_V4);
    w.write_str(sink_id);
    w.write_u64(issuer as u64);
    w.write_u64(floor);
    w.into_bytes()
}

pub fn fence_ack_frame(
    sink_id: &str,
    issuer: usize,
    applied_floor: u64,
    challenge_nonce: u64,
) -> Vec<u8> {
    let mut w = CanonicalWriter::with_domain(DOMAIN_FENCE_ACK_V4);
    w.write_str(sink_id);
    w.write_u64(issuer as u64);
    w.write_u64(applied_floor);
    w.write_u64(challenge_nonce);
    w.into_bytes()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedFenceCert {
    pub sink_id: String,
    pub issuer: usize,
    pub floor: u64,
    pub signature: Sig64,
}

impl SignedFenceCert {
    pub fn sign(
        issuer_kp: &Ed25519Keypair,
        sink_id: impl Into<String>,
        issuer: usize,
        floor: u64,
    ) -> Self {
        let sink_id = sink_id.into();
        let frame = fence_cert_frame(&sink_id, issuer, floor);
        let signature = issuer_kp.sign(&frame);
        Self {
            sink_id,
            issuer,
            floor,
            signature,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedFenceAck {
    pub sink_id: String,
    pub issuer: usize,
    pub applied_floor: u64,
    pub challenge_nonce: u64,
    pub signature: Sig64,
}

impl SignedFenceAck {
    pub fn verify(
        &self,
        expected_sink_id: &str,
        expected_sink_pk: &[u8; 32],
        expected_issuer: usize,
        required_floor: u64,
        expected_challenge_nonce: u64,
    ) -> bool {
        if self.sink_id != expected_sink_id
            || self.issuer != expected_issuer
            || self.applied_floor < required_floor
            || self.challenge_nonce != expected_challenge_nonce
        {
            return false;
        }
        let frame = fence_ack_frame(
            &self.sink_id,
            self.issuer,
            self.applied_floor,
            self.challenge_nonce,
        );
        ed25519_verify(expected_sink_pk, &frame, &self.signature.0)
    }
}

// ============================================================================
// Signed Partitioned-Budget Lease Slices (`jeath.lease.v4`)
// ============================================================================

pub fn lease_slice_frame(
    sink_id: &str,
    lease_id: u64,
    slice_budget: u64,
    expires_ms: u64,
) -> Vec<u8> {
    let mut w = CanonicalWriter::with_domain(DOMAIN_LEASE_V4);
    w.write_str(sink_id);
    w.write_u64(lease_id);
    w.write_u64(slice_budget);
    w.write_u64(expires_ms);
    w.into_bytes()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignedLeaseSlice {
    pub sink_id: String,
    pub lease_id: u64,
    pub slice_budget: u64,
    pub expires_ms: u64,
    pub signature: Sig64,
}

impl SignedLeaseSlice {
    pub fn sign(
        broker_kp: &Ed25519Keypair,
        sink_id: impl Into<String>,
        lease_id: u64,
        slice_budget: u64,
        expires_ms: u64,
    ) -> Self {
        let sink_id = sink_id.into();
        let frame = lease_slice_frame(&sink_id, lease_id, slice_budget, expires_ms);
        Self {
            sink_id,
            lease_id,
            slice_budget,
            expires_ms,
            signature: broker_kp.sign(&frame),
        }
    }

    pub fn verify(&self, expected_sink_id: &str, broker_pk: &[u8; 32]) -> bool {
        if self.sink_id != expected_sink_id {
            return false;
        }
        let frame = lease_slice_frame(
            &self.sink_id,
            self.lease_id,
            self.slice_budget,
            self.expires_ms,
        );
        ed25519_verify(broker_pk, &frame, &self.signature.0)
    }
}

// ============================================================================
// Cryptographic Sink Verifier State
// ============================================================================

#[derive(Clone, Debug)]
pub struct CryptoSinkVerifier {
    pub sink_id: String,
    pub sink_keypair: Ed25519Keypair,
    pub issuer_pks: Vec<[u8; 32]>,
    pub macaroon_key: [u8; 32],
    pub floors: Vec<u64>,
}

impl CryptoSinkVerifier {
    pub fn new(
        sink_id: impl Into<String>,
        sink_keypair: Ed25519Keypair,
        issuer_pks: Vec<[u8; 32]>,
        macaroon_key: [u8; 32],
    ) -> Self {
        let n = issuer_pks.len();
        Self {
            sink_id: sink_id.into(),
            sink_keypair,
            issuer_pks,
            macaroon_key,
            floors: vec![0; n],
        }
    }

    pub fn apply_signed_fence(
        &mut self,
        cert: &SignedFenceCert,
        challenge_nonce: u64,
    ) -> Result<SignedFenceAck, String> {
        if cert.sink_id != self.sink_id {
            return Err("cross-sink fence certificate rejected".into());
        }
        let Some(issuer_pk) = self.issuer_pks.get(cert.issuer) else {
            return Err("unknown issuer index".into());
        };
        let frame = fence_cert_frame(&cert.sink_id, cert.issuer, cert.floor);
        if !ed25519_verify(issuer_pk, &frame, &cert.signature.0) {
            return Err("invalid Ed25519 signature on fence certificate".into());
        }
        self.floors[cert.issuer] = policy_core::advance_floor(self.floors[cert.issuer], cert.floor);
        let applied_floor = self.floors[cert.issuer];
        let ack_bytes = fence_ack_frame(&self.sink_id, cert.issuer, applied_floor, challenge_nonce);
        let signature = self.sink_keypair.sign(&ack_bytes);
        Ok(SignedFenceAck {
            sink_id: self.sink_id.clone(),
            issuer: cert.issuer,
            applied_floor,
            challenge_nonce,
            signature,
        })
    }

    pub fn verify_quorum_authorization(
        &self,
        proposal: &CanonicalProposal,
        session_actor: &str,
        cert: &QuorumCertificate,
        now: u64,
    ) -> Result<Vec<bool>, String> {
        let n = self.issuer_pks.len();
        if proposal.sink_id != self.sink_id || cert.sink_id != self.sink_id {
            return Err("sink_id mismatch".into());
        }
        if proposal.actor != session_actor || proposal.id.is_empty() || proposal.id.len() > 128 {
            return Err("session actor or proposal id invalid".into());
        }
        if !proposal.effect.scoped() || now >= proposal.expires {
            return Err("proposal out of scope or expired".into());
        }
        if proposal.epochs.len() != n || cert.required.len() != n {
            return Err("issuer vector length mismatch".into());
        }
        let base_req = proposal.effect.base_required(n);
        if !policy_core::grants_ok(&base_req, &cert.required) {
            return Err("quorum certificate weakens required effect quorum".into());
        }
        let digest = proposal.digest();
        if !ct_eq_32(&digest, &cert.proposal_digest) {
            return Err("proposal digest mismatch".into());
        }
        if !policy_core::unfenced(&cert.required, &proposal.epochs, &self.floors) {
            return Err("proposal epoch fenced at sink".into());
        }
        for (i, &need) in cert.required.iter().enumerate() {
            if !need {
                continue;
            }
            let Some(vote) = cert.votes.iter().find(|v| v.issuer == i) else {
                return Err(format!("missing quorum signature for issuer {i}"));
            };
            if vote.epoch != proposal.epochs[i] {
                return Err(format!("vote epoch mismatch for issuer {i}"));
            }
            let frame = quorum_vote_frame(&self.sink_id, &digest, i, vote.epoch);
            if !ed25519_verify(&self.issuer_pks[i], &frame, &vote.signature.0) {
                return Err(format!("invalid Ed25519 quorum signature for issuer {i}"));
            }
        }
        Ok(cert.required.clone())
    }

    pub fn verify_macaroon_authorization(
        &self,
        proposal: &CanonicalProposal,
        session_actor: &str,
        macaroon: &CapabilityMacaroon,
        now: u64,
    ) -> Result<VerifiedCapability, String> {
        let n = self.issuer_pks.len();
        if proposal.sink_id != self.sink_id || macaroon.root.sink_id != self.sink_id {
            return Err("macaroon sink_id mismatch".into());
        }
        if proposal.actor != session_actor || proposal.id.is_empty() || proposal.id.len() > 128 {
            return Err("session actor or proposal id invalid".into());
        }
        if !proposal.effect.scoped() {
            return Err("proposal effect out of scope".into());
        }
        let cap = macaroon.verify(&self.macaroon_key)?;
        if cap.holder != session_actor {
            return Err("macaroon holder mismatch".into());
        }
        if now >= cap.expires || now >= proposal.expires || proposal.expires > cap.expires {
            return Err("macaroon or proposal expired".into());
        }
        let amount = proposal.effect.charge();
        if amount == 0 || amount > cap.budget {
            return Err("proposal exceeds attenuated macaroon budget".into());
        }
        if cap.scope.get(proposal.effect.kind_index()) != Some(&true) {
            return Err("proposal effect kind disallowed by macaroon scope".into());
        }
        let base_req = proposal.effect.base_required(n);
        if cap.required.len() != n
            || proposal.epochs.len() != n
            || cap.epochs != proposal.epochs
            || !policy_core::grants_ok(&base_req, &cap.required)
        {
            return Err("macaroon issuer/epoch requirements invalid".into());
        }
        if !policy_core::unfenced(&cap.required, &cap.epochs, &self.floors) {
            return Err("macaroon capability fenced at sink".into());
        }
        if let Some(pinned) = cap.pinned_proposal_digest
            && !ct_eq_32(&pinned, &proposal.digest())
        {
            return Err("proposal digest does not match pinned macaroon digest".into());
        }
        Ok(cap)
    }
}

// ============================================================================
// Exhaustive 10,000-Trial Red-Team & Fuzz Campaign
// ============================================================================

#[derive(Clone, Debug)]
struct LocalRng(u64);

impl LocalRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let limit = u64::MAX - u64::MAX % (n as u64);
        loop {
            let x = self.next_u64();
            if x < limit {
                return (x % (n as u64)) as usize;
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedTeamAttackBreakdown {
    pub family: String,
    pub attempts: usize,
    pub false_accepts: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoRedTeamReport {
    pub seed: u64,
    pub total_attacks: usize,
    pub false_accepts: usize,
    pub positive_controls_checked: usize,
    pub positive_controls_passed: usize,
    pub families: Vec<RedTeamAttackBreakdown>,
}

pub fn run_crypto_redteam_campaign(seed: u64, total_attacks: usize) -> CryptoRedTeamReport {
    let mut rng = LocalRng(seed);
    let issuers: Vec<Ed25519Keypair> = (0..3)
        .map(|i| Ed25519Keypair::deterministic("redteam.issuer", i))
        .collect();
    let attacker_kp = Ed25519Keypair::deterministic("redteam.attacker", 99);
    let sink0_kp = Ed25519Keypair::deterministic("redteam.sink", 0);
    let sink1_kp = Ed25519Keypair::deterministic("redteam.sink", 1);
    let issuer_pks: Vec<[u8; 32]> = issuers.iter().map(|k| k.public_key).collect();

    let mut mac_key_0 = [0u8; 32];
    let mut mac_key_1 = [0u8; 32];
    mac_key_0.copy_from_slice(&sha256(b"jeath.redteam.macaroon.sink0"));
    mac_key_1.copy_from_slice(&sha256(b"jeath.redteam.macaroon.sink1"));

    let family_names = [
        "quorum_sig_bitflip",
        "proposal_payload_tamper",
        "cross_sink_quorum_replay",
        "canonical_framing_boundary_splice",
        "missing_or_wrong_issuer_quorum",
        "macaroon_caveat_truncation",
        "macaroon_offline_attenuation_inflation",
        "macaroon_tag_or_root_bitflip",
        "forged_dos_fence_certificate",
        "mitm_stripped_or_stale_fence_ack",
    ];
    let mut attempts = [0usize; 10];
    let mut false_accepts = [0usize; 10];
    let mut positive_passed = 0usize;

    let sink0 = CryptoSinkVerifier::new("sink-0", sink0_kp.clone(), issuer_pks.clone(), mac_key_0);
    let sink1 = CryptoSinkVerifier::new("sink-1", sink1_kp.clone(), issuer_pks.clone(), mac_key_1);
    let key_refs: Vec<Option<&Ed25519Keypair>> = issuers.iter().map(Some).collect();

    // Pre-generate 64 distinct signed quorum fixtures (each positively verified) and verify
    // fresh HMAC macaroon attenuation on every iteration so debug-mode runtime stays < 10s.
    let pool_len = 64usize;
    let mut qc_pool = Vec::with_capacity(pool_len);
    for slot in 0..pool_len {
        let base_prop = CanonicalProposal {
            id: format!("prop-pool-{slot}"),
            actor: "alice".into(),
            sink_id: "sink-0".into(),
            epochs: vec![0, 0, 0],
            expires: 1_000,
            effect: CanonicalEffect::Release {
                destination: "approved".into(),
                contents: "SYNTHETIC DATA: no personal information".into(),
            },
        };
        let valid_qc =
            QuorumCertificate::sign_for_proposal(&base_prop, vec![true, true, true], &key_refs);
        assert!(
            sink0
                .verify_quorum_authorization(&base_prop, "alice", &valid_qc, 100)
                .is_ok()
        );
        qc_pool.push((base_prop, valid_qc));
    }

    for trial in 0..total_attacks {
        let (base_prop, valid_qc) = &qc_pool[trial % pool_len];

        let root_mac = CapabilityMacaroon::mint(
            &mac_key_0,
            MacaroonRoot {
                root_id: format!("root-{trial}"),
                sink_id: "sink-0".into(),
                initial_holder: "alice".into(),
                budget: 8,
                expires: 1_200,
                scope: vec![true, true, false],
                required: vec![true, true, true],
                epochs: vec![0, 0, 0],
            },
        )
        .expect("valid root macaroon");
        let attenuated_mac = root_mac
            .attenuate_offline(MacaroonCaveat {
                delegatee: "bob".into(),
                budget: 4,
                expires: 1_050,
                scope: vec![true, false, false],
                required: vec![true, true, true],
                pinned_proposal_digest: None,
            })
            .expect("valid offline caveat");
        let bob_prop = CanonicalProposal {
            id: format!("bob-{trial}"),
            actor: "bob".into(),
            sink_id: "sink-0".into(),
            epochs: vec![0, 0, 0],
            expires: 1_000,
            effect: CanonicalEffect::Transfer {
                recipient: "approved".into(),
                units: 4,
            },
        };

        if sink0
            .verify_macaroon_authorization(&bob_prop, "bob", &attenuated_mac, 100)
            .is_ok()
        {
            positive_passed += 1;
        }

        let fam = (trial + rng.index(10)) % 10;
        attempts[fam] += 1;

        let falsely_accepted = match fam {
            0 => {
                // Flip 1..=3 distinct bit positions in one of the Ed25519 quorum signatures
                let mut bad_qc = valid_qc.clone();
                let vote_idx = rng.index(bad_qc.votes.len());
                let flips = 1 + rng.index(3);
                let base_bit = rng.index(512);
                for k in 0..flips {
                    let pos = (base_bit + k * 137) % 512;
                    bad_qc.votes[vote_idx].signature.0[pos / 8] ^= 1u8 << (pos % 8);
                }
                debug_assert_ne!(bad_qc, *valid_qc);
                sink0
                    .verify_quorum_authorization(base_prop, "alice", &bad_qc, 100)
                    .is_ok()
            }
            1 => {
                // Mutate proposal units/expiry/actor/id after quorum signing
                let mut mutated = base_prop.clone();
                match rng.index(4) {
                    0 => mutated.expires += 1 + (rng.next_u64() % 500),
                    1 => mutated.id.push('x'),
                    2 => mutated.epochs[rng.index(3)] += 1,
                    _ => {
                        mutated.effect = CanonicalEffect::Transfer {
                            recipient: "approved".into(),
                            units: 8,
                        };
                    }
                }
                sink0
                    .verify_quorum_authorization(&mutated, "alice", valid_qc, 100)
                    .is_ok()
            }
            2 => {
                // Cross-sink replay: present Sink_0's valid certificate to Sink_1
                let mut cross_prop = base_prop.clone();
                if rng.index(2) == 0 {
                    cross_prop.sink_id = "sink-1".into();
                }
                sink1
                    .verify_quorum_authorization(&cross_prop, "alice", valid_qc, 100)
                    .is_ok()
            }
            3 => {
                // Canonical framing splice attack: shift boundary between adjacent strings
                // (e.g., id="ab", actor="cd" vs id="a", actor="bcd") and reuse signature!
                let p_left = CanonicalProposal {
                    id: format!("ab-{trial}"),
                    actor: "cd".into(),
                    sink_id: "sink-0".into(),
                    epochs: vec![0, 0, 0],
                    expires: 1_000,
                    effect: CanonicalEffect::Transfer {
                        recipient: "approved".into(),
                        units: 1,
                    },
                };
                let p_spliced = CanonicalProposal {
                    id: "a".to_string(),
                    actor: format!("b-{trial}cd"),
                    sink_id: "sink-0".into(),
                    epochs: vec![0, 0, 0],
                    expires: 1_000,
                    effect: CanonicalEffect::Transfer {
                        recipient: "approved".into(),
                        units: 1,
                    },
                };
                let qc_left = QuorumCertificate::sign_for_proposal(
                    &p_left,
                    vec![true, false, false],
                    &key_refs,
                );
                let mut qc_spliced = qc_left;
                if rng.index(2) == 0 {
                    // Attacker updates proposal_digest field to p_spliced.digest() while keeping signatures
                    qc_spliced.proposal_digest = p_spliced.digest();
                }
                sink0
                    .verify_quorum_authorization(&p_spliced, &p_spliced.actor, &qc_spliced, 100)
                    .is_ok()
            }
            4 => {
                // Missing required quorum issuer OR signature from attacker/wrong issuer key
                let mut bad_qc = valid_qc.clone();
                let target_issuer = rng.index(3);
                if rng.index(2) == 0 {
                    bad_qc.votes.retain(|v| v.issuer != target_issuer);
                } else {
                    let frame = quorum_vote_frame(
                        &base_prop.sink_id,
                        &base_prop.digest(),
                        target_issuer,
                        0,
                    );
                    let wrong_signer = if rng.index(2) == 0 {
                        &attacker_kp
                    } else {
                        &issuers[(target_issuer + 1) % 3]
                    };
                    bad_qc.votes[target_issuer].signature = wrong_signer.sign(&frame);
                }
                sink0
                    .verify_quorum_authorization(base_prop, "alice", &bad_qc, 100)
                    .is_ok()
            }
            5 => {
                // Macaroon caveat truncation: strip `bob` attenuation caveat while only knowing `attenuated_mac.tag`
                let mut stripped = attenuated_mac.clone();
                stripped.caveats.clear();
                if rng.index(2) == 1 {
                    let b_idx = rng.index(32);
                    stripped.tag[b_idx] ^= 1u8 << rng.index(8);
                }
                let alice_big = CanonicalProposal {
                    id: format!("alice-big-{trial}"),
                    actor: "alice".into(),
                    sink_id: "sink-0".into(),
                    epochs: vec![0, 0, 0],
                    expires: 1_000,
                    effect: CanonicalEffect::Transfer {
                        recipient: "approved".into(),
                        units: 8,
                    },
                };
                sink0
                    .verify_macaroon_authorization(&alice_big, "alice", &stripped, 100)
                    .is_ok()
            }
            6 => {
                // Offline attenuation inflation: Bob HAS `attenuated_mac` (budget=4, expires=1050, scope=[T,F,F], req=[T,T,T])
                // and computes a cryptographically valid chained HMAC tag for an inflated caveat!
                let inflated_caveat = match rng.index(4) {
                    0 => MacaroonCaveat {
                        delegatee: "mallory".into(),
                        budget: 5 + (rng.next_u64() % 10), // > 4
                        expires: 1_000,
                        scope: vec![true, false, false],
                        required: vec![true, true, true],
                        pinned_proposal_digest: None,
                    },
                    1 => MacaroonCaveat {
                        delegatee: "mallory".into(),
                        budget: 4,
                        expires: 1_051 + (rng.next_u64() % 200), // > 1050
                        scope: vec![true, false, false],
                        required: vec![true, true, true],
                        pinned_proposal_digest: None,
                    },
                    2 => MacaroonCaveat {
                        delegatee: "mallory".into(),
                        budget: 1,
                        expires: 1_000,
                        scope: vec![true, true, false], // widens scope[1] from false -> true
                        required: vec![true, true, true],
                        pinned_proposal_digest: None,
                    },
                    _ => MacaroonCaveat {
                        delegatee: "mallory".into(),
                        budget: 2,
                        expires: 1_000,
                        scope: vec![true, false, false],
                        required: vec![true, true, false], // drops required[2]
                        pinned_proposal_digest: None,
                    },
                };
                let forged_mac = attenuated_mac.append_caveat_raw_for_test(inflated_caveat);
                let mallory_prop = CanonicalProposal {
                    id: format!("mal-{trial}"),
                    actor: "mallory".into(),
                    sink_id: "sink-0".into(),
                    epochs: vec![0, 0, 0],
                    expires: 999,
                    effect: CanonicalEffect::Transfer {
                        recipient: "approved".into(),
                        units: 1,
                    },
                };
                forged_mac.verify(&mac_key_0).is_ok()
                    || sink0
                        .verify_macaroon_authorization(&mallory_prop, "mallory", &forged_mac, 100)
                        .is_ok()
            }
            7 => {
                // Macaroon tag bitflip or root budget tamper
                let mut tampered = attenuated_mac.clone();
                if rng.index(2) == 0 {
                    tampered.tag[rng.index(32)] ^= 1u8 << rng.index(8);
                } else {
                    tampered.root.budget += 10;
                }
                tampered.verify(&mac_key_0).is_ok()
            }
            8 => {
                // Forged DoS fence certificate: signed by attacker or wrong issuer or cross-sink
                let mut sink_mut = sink0.clone();
                let issuer_idx = rng.index(3);
                let bad_cert = match rng.index(3) {
                    0 => SignedFenceCert::sign(&attacker_kp, "sink-0", issuer_idx, 5),
                    1 => SignedFenceCert::sign(
                        &issuers[(issuer_idx + 1) % 3],
                        "sink-0",
                        issuer_idx,
                        5,
                    ),
                    _ => SignedFenceCert::sign(&issuers[issuer_idx], "sink-1", issuer_idx, 5),
                };
                sink_mut.apply_signed_fence(&bad_cert, 42).is_ok()
            }
            _ => {
                // MITM stripped fence / stale or wrong-nonce or wrong-sink SignedFenceAck
                let mut sink_mut = sink0.clone();
                let mut other_sink = sink1.clone();
                let valid_fence_low = SignedFenceCert::sign(&issuers[0], "sink-0", 0, 1);
                let ack_low = sink_mut
                    .apply_signed_fence(&valid_fence_low, 1001)
                    .expect("valid low fence");
                let valid_fence_s1 = SignedFenceCert::sign(&issuers[0], "sink-1", 0, 2);
                let ack_s1 = other_sink
                    .apply_signed_fence(&valid_fence_s1, 2002)
                    .expect("valid sink1 fence");

                match rng.index(4) {
                    0 => {
                        // Replay floor=1 ack when broker requires floor=2
                        ack_low.verify("sink-0", &sink0_kp.public_key, 0, 2, 1001)
                    }
                    1 => {
                        // Replay ack with stale challenge_nonce
                        let f2 = SignedFenceCert::sign(&issuers[0], "sink-0", 0, 2);
                        let ack2 = sink_mut.apply_signed_fence(&f2, 1001).expect("f2");
                        ack2.verify("sink-0", &sink0_kp.public_key, 0, 2, 2002)
                    }
                    2 => {
                        // Cross-sink FenceAck substitution (Sink_1's valid floor=2, nonce=2002 ack presented as Sink_0)
                        ack_s1.verify("sink-0", &sink0_kp.public_key, 0, 2, 2002)
                    }
                    _ => {
                        // Bit-flip on valid FenceAck signature
                        let f2 = SignedFenceCert::sign(&issuers[0], "sink-0", 0, 2);
                        let mut ack2 = sink_mut.apply_signed_fence(&f2, 2002).expect("f2");
                        ack2.signature.0[rng.index(64)] ^= 1u8 << rng.index(8);
                        ack2.verify("sink-0", &sink0_kp.public_key, 0, 2, 2002)
                    }
                }
            }
        };

        if falsely_accepted {
            false_accepts[fam] += 1;
        }
    }

    let families = family_names
        .iter()
        .enumerate()
        .map(|(i, &name)| RedTeamAttackBreakdown {
            family: name.into(),
            attempts: attempts[i],
            false_accepts: false_accepts[i],
        })
        .collect();

    CryptoRedTeamReport {
        seed,
        total_attacks,
        false_accepts: false_accepts.iter().sum(),
        positive_controls_checked: total_attacks,
        positive_controls_passed: positive_passed,
        families,
    }
}
