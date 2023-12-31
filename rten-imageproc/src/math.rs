use crate::Point;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn from_yx(y: f32, x: f32) -> Vec2 {
        Vec2 { y, x }
    }

    pub fn from_xy(x: f32, y: f32) -> Vec2 {
        Vec2 { x, y }
    }

    /// Return the vector from `start` to `end`.
    pub fn from_points(start: Point, end: Point) -> Vec2 {
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        Vec2::from_yx(dy as f32, dx as f32)
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Return the magnitude of the cross product of this vector with `other`.
    pub fn cross_product_norm(&self, other: Vec2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Return the dot product of this vector with `other`.
    pub fn dot(&self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Return a copy of this vector scaled such that the length is 1.
    pub fn normalized(&self) -> Vec2 {
        let inv_len = 1. / self.length();
        Vec2::from_yx(self.y * inv_len, self.x * inv_len)
    }

    /// Return a vector perpendicular to this vector.
    pub fn perpendicular(&self) -> Vec2 {
        Vec2 {
            y: -self.x,
            x: self.y,
        }
    }
}

impl std::ops::Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            y: self.y + rhs.y,
            x: self.x + rhs.x,
        }
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Vec2 {
        Vec2 {
            y: -self.y,
            x: -self.x,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f32) -> Vec2 {
        Vec2 {
            y: self.y * rhs,
            x: self.x * rhs,
        }
    }
}

impl std::ops::Sub<f32> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: f32) -> Vec2 {
        Vec2 {
            y: self.y - rhs,
            x: self.x - rhs,
        }
    }
}

impl std::ops::Sub<Vec2> for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            y: self.y - rhs.y,
            x: self.x - rhs.x,
        }
    }
}
