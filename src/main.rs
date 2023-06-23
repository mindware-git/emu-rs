use std::env;

const REG_SIZE: usize = 4;
struct Register {
    pc: u32,
    x: [u32; REG_SIZE],
}
impl Register {
    fn init(&mut self) {
        self.pc = 0;
        for iter in &mut self.x {
            *iter = 0;
        }
    }
}
fn main() {
    println!("Put hex dump file");
    let mut reg: Register = Register {
        pc: (1),
        x: [2, 3, 4, 5],
    };
    reg.init();

    let args: Vec<String> = env::args().collect();
    assert!(args.len() == 2);
}
#[cfg(test)]
mod tests {
    use crate::Register;

    #[test]
    fn reg_init() {
        let mut reg: Register = Register {
            pc: (1),
            x: [2, 3, 4, 5],
        };
        reg.init();

        assert_eq!(reg.pc, 0);
        for iter in reg.x {
            assert_eq!(iter, 0);
        }
    }
}
