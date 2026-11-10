//! Namespace-independent Linux proposer confinement. The launcher supplies only
//! stdio RPC, no database, sink, administrator, or credential descriptors.
use std::{io, path::Path};

pub fn landlock_abi() -> i64 {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<u8>(),
                0,
                1,
            )
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

#[cfg(target_os = "linux")]
pub fn confine(executable: &Path) -> io::Result<u32> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    #[repr(C)]
    struct Ruleset {
        handled: u64,
    }
    #[repr(C, packed)]
    struct Beneath {
        allowed: u64,
        fd: i32,
    }
    let error = || io::Error::last_os_error();
    // ABI >= 4 supplies all filesystem rights through TRUNCATE and TCP rights.
    let abi = landlock_abi();
    if abi < 4 {
        return Err(io::Error::other(format!(
            "Landlock ABI >=4 required; got {abi}"
        )));
    }
    let all_fs = (1u64 << 15) - 1;
    let rules = Ruleset { handled: all_fs };
    let fd = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &rules,
            std::mem::size_of::<Ruleset>(),
            0,
        )
    } as i32;
    if fd < 0 {
        return Err(error());
    }
    let result = (|| {
        for (path, allowed) in [
            (Path::new("/usr"), 1 | 4 | 8),
            (Path::new("/lib"), 1 | 4 | 8),
            (Path::new("/lib64"), 1 | 4 | 8),
            (Path::new("/bin"), 1 | 4 | 8),
            (Path::new("/etc/ld.so.cache"), 4),
            (Path::new("/dev/null"), 2 | 4),
            (Path::new("/dev/urandom"), 4),
            (executable, 1 | 4),
        ] {
            if !path.exists() {
                continue;
            }
            let name = CString::new(path.as_os_str().as_bytes())?;
            let pfd = unsafe { libc::open(name.as_ptr(), libc::O_PATH | libc::O_CLOEXEC) };
            if pfd < 0 {
                return Err(error());
            }
            let rule = Beneath { allowed, fd: pfd };
            let rc = unsafe { libc::syscall(libc::SYS_landlock_add_rule, fd, 1, &rule, 0) };
            unsafe {
                libc::close(pfd);
            }
            if rc < 0 {
                return Err(error());
            }
        }
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            return Err(error());
        }
        if unsafe { libc::syscall(libc::SYS_landlock_restrict_self, fd, 0) } != 0 {
            return Err(error());
        }
        Ok(())
    })();
    unsafe {
        libc::close(fd);
    }
    result?;
    // Architecture check also blocks x32/compat syscall-number bypasses.
    #[cfg(target_arch = "x86_64")]
    let arch = 0xc000003eu32;
    #[cfg(target_arch = "aarch64")]
    let arch = 0xc00000b7u32;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    compile_error!("v2 confinement requires an audited seccomp architecture");
    let ins = |code, jt, jf, k| libc::sock_filter { code, jt, jf, k };
    let mut filter = vec![
        ins(0x20, 0, 0, 4),
        ins(0x15, 1, 0, arch),
        ins(0x06, 0, 0, 0x80000000),
        ins(0x20, 0, 0, 0),
        ins(0x35, 0, 1, 0x40000000),
        ins(0x06, 0, 0, 0x80000000),
    ];
    // prlimit(0, ...) can inspect this process; a nonzero PID could alter another
    // process owned by the shared host user.
    filter.extend([
        ins(0x15, 0, 3, libc::SYS_prlimit64 as u32),
        ins(0x20, 0, 0, 16),
        ins(0x15, 1, 0, 0),
        ins(0x06, 0, 0, 0x00050000 | libc::EPERM as u32),
        ins(0x20, 0, 0, 0),
    ]);
    for nr in [
        libc::SYS_socket,
        libc::SYS_socketpair,
        libc::SYS_connect,
        libc::SYS_bind,
        libc::SYS_listen,
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_sendto,
        libc::SYS_sendmsg,
        libc::SYS_sendmmsg,
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_pidfd_getfd,
        libc::SYS_pidfd_open,
        libc::SYS_pidfd_send_signal,
        libc::SYS_kill,
        libc::SYS_tkill,
        libc::SYS_tgkill,
        libc::SYS_rt_sigqueueinfo,
        libc::SYS_rt_tgsigqueueinfo,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_unshare,
        libc::SYS_setns,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_userfaultfd,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_open_by_handle_at,
        libc::SYS_reboot,
        libc::SYS_kexec_load,
    ] {
        filter.push(ins(0x15, 0, 1, nr as u32));
        filter.push(ins(0x06, 0, 0, 0x00050000 | libc::EPERM as u32));
    }
    filter.push(ins(0x06, 0, 0, 0x7fff0000));
    let prog = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    if unsafe { libc::prctl(libc::PR_SET_SECCOMP, 2, &prog) } != 0 {
        return Err(error());
    }
    Ok(abi as u32)
}
#[cfg(not(target_os = "linux"))]
pub fn confine(_: &Path) -> io::Result<u32> {
    Err(io::Error::other(
        "v2 confinement requires Linux Landlock and seccomp",
    ))
}

pub fn probe(database: &Path, parent: u32) -> serde_json::Value {
    let db = std::fs::read(database).is_ok();
    let db_write = std::fs::OpenOptions::new()
        .write(true)
        .open(database)
        .is_ok();
    let inspect = std::fs::read(format!("/proc/{parent}/environ")).is_ok();
    #[cfg(target_os = "linux")]
    let (network, trace, vm_read) = unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
        if fd >= 0 {
            libc::close(fd);
        }
        let marker = 42u64;
        let child = libc::fork();
        if child == 0 {
            libc::usleep(100_000);
            libc::_exit(0);
        }
        let mut value = 0u64;
        let local = libc::iovec {
            iov_base: (&mut value as *mut u64).cast(),
            iov_len: 8,
        };
        let remote = libc::iovec {
            iov_base: (&marker as *const u64).cast_mut().cast(),
            iov_len: 8,
        };
        let read = child > 0
            && libc::process_vm_readv(child, &local, 1, &remote, 1, 0) == 8
            && value == 42;
        let rc = if child > 0 {
            libc::ptrace(libc::PTRACE_ATTACH, child, 0, 0)
        } else {
            -1
        };
        if rc == 0 {
            libc::waitpid(child, std::ptr::null_mut(), 0);
            libc::ptrace(libc::PTRACE_DETACH, child, 0, 0);
        }
        if child > 0 {
            libc::waitpid(child, std::ptr::null_mut(), 0);
        }
        (fd >= 0, rc == 0, read)
    };
    #[cfg(not(target_os = "linux"))]
    let (network, trace, vm_read) = (
        std::net::TcpListener::bind("127.0.0.1:0").is_ok(),
        false,
        false,
    );
    serde_json::json!({"database_read":db,"database_write":db_write,"proc_read":inspect,"socket":network,"ptrace":trace,"process_vm_read":vm_read})
}
