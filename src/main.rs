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

fn decode(hex_str: &str) -> u32 {
    let inst: u32 = u32::from_str_radix(hex_str, 16).expect("gogo");
    inst
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
    use std::{fs, path::Path};

    use crate::{decode, Register};

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

    #[test]
    fn read_hex() {
        let file_path = Path::new("src/simple1.hex");
        println!("In file {}", file_path.display());

        let contents =
            fs::read_to_string(file_path).expect("Should have been able to read the file");

        for line in contents.lines() {
            println!("{}", line);
            let inst = decode(line);
            println!("binary form {inst:b}");
        }
    }
}
