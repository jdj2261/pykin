class A:
    @property
    def attr(self):
        try:
            return self._attr
        except AttributeError:
            return ''

class B(A):
    @property
    def attr(self):
        return super(B, self).attr

    @attr.setter
    def attr(self, value):
        self._attr = value

if __name__ == '__main__':
    b = B()
    print('Before set:', repr(b.attr))
    b.attr = 'abc'
    print(' After set:', repr(b.attr))
