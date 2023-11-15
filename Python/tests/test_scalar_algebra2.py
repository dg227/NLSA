import numpy as np
from fractions import Fraction
from nlsa.abstract_algebra2 import FromScalarField,\
        ImplementsUnitalStarAlgebraLRModule, ImplementsPower, ImplementsSqrt
from nlsa.scalar_algebra2 import IntField, IntegerField, RationalField,\
        FloatField, FloatingField, ComplexField, ComplexfloatingField
from numpy.testing import assert_almost_equal


def test_intfield():
    scl = FromScalarField(IntField)
    x1 = int(1)
    x2 = int(2)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert isinstance(scl, ImplementsPower)
    assert scl.add(x1, x2) == int(3)
    assert scl.sub(x1, x2) == int(-1)
    assert scl.neg(x1) == -1
    assert scl.mul(x1, x2) == int(2)
    assert scl.smul(x1, x2) == int(2)
    assert scl.power(x2, x2) == int(4)
    assert scl.lmul(x1, x2) == int(2)
    assert scl.rmul(x1, x2) == int(2)
    assert scl.div(x1, x2) == int(0)
    assert scl.inv(x2) == int(0)
    assert scl.ldiv(x1, x2) == int(2)
    assert scl.rdiv(x1, x2) == int(0)
    assert scl.unit() == int(1)
    assert scl.star(x2) == int(2)


def test_integerfield():
    scl = FromScalarField(IntegerField(np.int32))
    x1 = np.int32(1)
    x2 = np.int32(2)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert isinstance(scl, ImplementsPower)
    assert scl.add(x1, x2) == np.int32(3)
    assert scl.sub(x1, x2) == np.int32(-1)
    assert scl.neg(x1) == np.int32(-1)
    assert scl.mul(x1, x2) == np.int32(2)
    assert scl.smul(x1, x2) == np.int32(2)
    assert scl.power(x2, x2) == np.int32(4)
    assert scl.lmul(x1, x2) == np.int32(2)
    assert scl.rmul(x1, x2) == np.int32(2)
    assert scl.div(x1, x2) == np.int32(0)
    assert scl.inv(x2) == np.int32(0)
    assert scl.ldiv(x1, x2) == np.int32(2)
    assert scl.rdiv(x1, x2) == np.int32(0)
    assert scl.unit() == np.int32(1)
    assert scl.star(x2) == np.int32(2)


def test_rationalfield():
    scl = FromScalarField(RationalField)
    x1 = int(1)
    x2 = int(2)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert scl.add(x1, x2) == int(3)
    assert scl.sub(x1, x2) == int(-1)
    assert scl.neg(x1) == int(-1)
    assert scl.mul(x1, x2) == int(2)
    assert scl.smul(x1, x2) == int(2)
    assert scl.lmul(x1, x2) == int(2)
    assert scl.rmul(x1, x2) == int(2)
    assert scl.div(x1, x2) == Fraction(1, 2)
    assert scl.inv(x2) == Fraction(1, 2)
    assert scl.ldiv(x1, x2) == int(2)
    assert scl.rdiv(x1, x2) == Fraction(1, 2)
    assert scl.unit() == int(1)
    assert scl.star(x2) == int(2)


def test_floatfield():
    scl = FromScalarField(FloatField)
    x1 = int(1)
    x2 = int(2)
    x4 = int(4)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert scl.add(x1, x2) == int(3)
    assert scl.sub(x1, x2) == int(-1)
    assert scl.neg(x1) == int(-1)
    assert scl.mul(x1, x2) == int(2)
    assert scl.smul(x1, x2) == int(2)
    assert scl.lmul(x1, x2) == int(2)
    assert scl.rmul(x1, x2) == int(2)
    assert_almost_equal(scl.div(x1, x2), float(0.5))
    assert_almost_equal(scl.inv(x2), float(0.5))
    assert scl.ldiv(x1, x2) == int(2)
    assert_almost_equal(scl.rdiv(x1, x2), float(0.5))
    assert_almost_equal(scl.sqrt(x4), 2)
    assert scl.unit() == int(1)
    assert scl.star(x2) == int(2)


def test_floatingfield():
    scl = FromScalarField(FloatingField(np.float32))
    x1 = np.float32(1)
    x2 = np.float32(2)
    x4 = np.float32(4)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert scl.add(x1, x2) == np.float32(3)
    assert scl.sub(x1, x2) == np.float32(-1)
    assert scl.neg(x1) == np.float32(-1)
    assert scl.mul(x1, x2) == np.float32(2)
    assert scl.smul(x1, x2) == np.float32(2)
    assert scl.lmul(x1, x2) == np.float32(2)
    assert scl.rmul(x1, x2) == np.float32(2)
    assert_almost_equal(scl.div(x1, x2), np.float32(0.5))
    assert_almost_equal(scl.inv(x2), np.float32(0.5))
    assert scl.ldiv(x1, x2) == np.float32(2)
    assert_almost_equal(scl.rdiv(x1, x2), np.float32(0.5))
    assert_almost_equal(scl.sqrt(x4), 2)
    assert scl.unit() == np.float32(1)
    assert scl.star(x2) == np.float32(2)


def test_complexfield():
    scl = FromScalarField(ComplexField)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert isinstance(scl, ImplementsSqrt)
    assert isinstance(scl, ImplementsPower)
    assert scl.add(1, 2) == 3
    assert scl.sub(1, 2) == -1
    assert scl.neg(1) == -1
    assert scl.mul(1, 2) == 2
    assert scl.power(2, 2) == 4
    assert scl.smul(1, 2) == 2
    assert scl.lmul(1, 2) == 2
    assert scl.rmul(1, 2) == 2
    assert_almost_equal(scl.div(1, 2), 0.5)
    assert_almost_equal(scl.inv(2), 0.5)
    assert scl.ldiv(1, 2) == 2
    assert_almost_equal(scl.rdiv(1, 2), 0.5)
    assert_almost_equal(scl.sqrt(4), 2)
    assert scl.unit() == 1
    assert scl.star(2j) == -2j


def test_complexfloatingfield():
    scl = FromScalarField(ComplexfloatingField(np.complex128))
    x1 = np.complex128(1)
    x2 = np.complex128(2)
    x2j = np.complex128(2j)
    x4 = np.complex128(4)
    assert isinstance(scl, ImplementsUnitalStarAlgebraLRModule)
    assert isinstance(scl, ImplementsSqrt)
    assert isinstance(scl, ImplementsPower)
    assert scl.add(x1, x2) == np.complex128(3)
    assert scl.sub(x1, x2) == np.complex128(-1)
    assert scl.neg(x1) == np.complex128(-1)
    assert scl.mul(x1, x2) == np.complex128(x2)
    assert scl.power(x2, x2) == np.complex128(4)
    assert scl.smul(x1, x2) == np.complex128(x2)
    assert scl.lmul(x1, x2) == np.complex128(x2)
    assert scl.rmul(x1, x2) == np.complex128(x2)
    assert_almost_equal(scl.div(x1, x2), 0.5)
    assert_almost_equal(scl.inv(x2), 0.5)
    assert scl.ldiv(x1, x2) == np.complex128(x2)
    assert_almost_equal(scl.rdiv(x1, x2), 0.5)
    assert_almost_equal(scl.sqrt(x4), x2)
    assert scl.unit() == np.complex128(x1)
    assert scl.star(x2j) == np.complex128(-2j)
