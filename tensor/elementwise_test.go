package tensor

import (
	"testing"
)

// Scalar references used to validate the SIMD elementwise kernels.

func scalarAdd(dst, a, b []float32) {
	for i := range a {
		dst[i] = a[i] + b[i]
	}
}

func scalarSub(dst, a, b []float32) {
	for i := range a {
		dst[i] = a[i] - b[i]
	}
}

func scalarMul(dst, a, b []float32) {
	for i := range a {
		dst[i] = a[i] * b[i]
	}
}

func scalarScale(dst, a []float32, s float32) {
	for i := range a {
		dst[i] = a[i] * s
	}
}

func scalarAddScaled(dst, a, b []float32, s float32) {
	for i := range a {
		dst[i] = a[i] + s*b[i]
	}
}

func scalarRelu2(dst, a []float32) {
	for i, v := range a {
		if v < 0 {
			v = 0
		}
		dst[i] = v * v
	}
}

func scalarSquared(dst, a []float32) {
	for i, v := range a {
		dst[i] = v * v
	}
}

func testData(n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(i%7) - 3.5 + 0.1*float32(i%3)
	}
	return d
}

func testDataB(n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(i%5) + 0.5
	}
	return d
}

func TestAddKernelParity(t *testing.T) {
	a := testData(1031)
	b := testDataB(1031)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	addKernel(got, a, b)
	scalarAdd(want, a, b)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("Add mismatch at %d: %v != %v", i, got[i], want[i])
		}
	}
}

func TestSubKernelParity(t *testing.T) {
	a := testData(1000)
	b := testDataB(1000)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	subKernel(got, a, b)
	scalarSub(want, a, b)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("Sub mismatch at %d", i)
		}
	}
}

func TestMulKernelParity(t *testing.T) {
	a := testData(513)
	b := testDataB(513)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	mulKernel(got, a, b)
	scalarMul(want, a, b)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("Mul mismatch at %d", i)
		}
	}
}

func TestScaleKernelParity(t *testing.T) {
	a := testData(77)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	scaleKernel(got, a, 2.5)
	scalarScale(want, a, 2.5)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("Scale mismatch at %d", i)
		}
	}
}

func TestAddScaledKernelParity(t *testing.T) {
	a := testData(2049)
	b := testDataB(2049)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	addScaledKernel(got, a, b, 1.5)
	scalarAddScaled(want, a, b, 1.5)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("AddScaled mismatch at %d", i)
		}
	}
}

func TestRelu2KernelParity(t *testing.T) {
	a := testData(3000)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	relu2Kernel(got, a)
	scalarRelu2(want, a)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("Relu2 mismatch at %d: %v != %v", i, got[i], want[i])
		}
	}
}

func TestSquaredKernelParity(t *testing.T) {
	a := testData(300)
	got := make([]float32, len(a))
	want := make([]float32, len(a))
	squaredKernel(got, a)
	scalarSquared(want, a)
	for i := range a {
		if got[i] != want[i] {
			t.Fatalf("Squared mismatch at %d", i)
		}
	}
}

func TestNegAbsParity(t *testing.T) {
	a := testData(640)
	neg := Neg(NewWithData([]int{640}, a))
	abs := Abs(NewWithData([]int{640}, a))
	for i := range a {
		if neg.Data[i] != -a[i] {
			t.Fatalf("Neg mismatch at %d", i)
		}
		want := a[i]
		if want < 0 {
			want = -want
		}
		if abs.Data[i] != want {
			t.Fatalf("Abs mismatch at %d", i)
		}
	}
}
